import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import FixedWindows, load_splits
from .metrics import masked_per_video_error, per_video_consistency, per_video_horizon_mse
from .models import build_model, one_step_reach, parameter_count
from .symmetrize import SymmetrizedForecaster
from .transforms import apply_group


BATCH_SIZE = 32


class BlankPredictor(nn.Module):
    def forward(self, inputs):
        return inputs.new_zeros((inputs.shape[0], 5, 1, 64, 64))


class CopyLastPredictor(nn.Module):
    def forward(self, inputs):
        return inputs[:, -1:].expand(-1, 5, -1, -1, -1)


def loader_options(workers):
    options = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 4
    return options


def make_loader(videos, speed, workers):
    return DataLoader(
        FixedWindows(videos, speed=speed, start=1),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_options(workers),
    )


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def mean_value(values):
    return float(torch.cat(values).mean())


def evaluate_predictor(predictor, loaders, device, name, timed=True):
    predictor.eval()
    original_horizon = []
    original_e5 = []
    original_e2 = []
    mirror_consistency = []
    mirror_error = []
    rotation_consistency = []
    rotation_error = []
    indices = []
    inference_seconds = 0.0
    with torch.no_grad():
        progress = tqdm(loaders[1], desc=f"{name} original and transforms", dynamic_ncols=True, leave=True, file=sys.stdout)
        for batch in progress:
            inputs = batch["inputs"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            synchronize(device)
            start = time.perf_counter()
            predictions = predictor(inputs)
            synchronize(device)
            if timed:
                inference_seconds += time.perf_counter() - start
            horizon = per_video_horizon_mse(predictions.float(), targets)
            original_horizon.append(horizon.cpu())
            original_e5.append(horizon.mean(dim=1).cpu())
            original_e2.append(horizon[:, :2].mean(dim=1).cpu())
            mirror_inputs = apply_group(inputs, 4)
            mirror_targets = apply_group(targets, 4)
            mirror_predictions = predictor(mirror_inputs)
            mirror_reference = apply_group(predictions, 4)
            mirror_consistency.append(per_video_consistency(mirror_predictions.float(), mirror_reference.float()).cpu())
            mirror_error.append(masked_per_video_error(mirror_predictions.float(), mirror_targets, mask).cpu())
            rotation_inputs = apply_group(inputs, 1)
            rotation_targets = apply_group(targets, 1)
            rotation_predictions = predictor(rotation_inputs)
            rotation_reference = apply_group(predictions, 1)
            rotation_consistency.append(per_video_consistency(rotation_predictions.float(), rotation_reference.float()).cpu())
            rotation_error.append(masked_per_video_error(rotation_predictions.float(), rotation_targets, mask).cpu())
            indices.append(batch["video_index"].cpu())
    speed_metrics = {}
    for speed in (2, 3):
        errors_e2 = []
        errors_e5 = []
        with torch.no_grad():
            progress = tqdm(loaders[speed], desc=f"{name} speed {speed}x", dynamic_ncols=True, leave=True, file=sys.stdout)
            for batch in progress:
                inputs = batch["inputs"].to(device, non_blocking=True)
                targets = batch["targets"].to(device, non_blocking=True)
                mask = batch["mask"].to(device, non_blocking=True)
                predictions = predictor(inputs)
                errors_e2.append(masked_per_video_error(predictions.float(), targets, mask, horizons=2).cpu())
                errors_e5.append(masked_per_video_error(predictions.float(), targets, mask).cpu())
        speed_metrics[str(speed)] = {
            "E2": mean_value(errors_e2),
            "E5": mean_value(errors_e5),
        }
    horizon_values = torch.cat(original_horizon)
    per_video = {
        "video_index": torch.cat(indices).numpy(),
        "horizon_mse": horizon_values.numpy(),
        "E5": torch.cat(original_e5).numpy(),
        "E2": torch.cat(original_e2).numpy(),
        "C_M": torch.cat(mirror_consistency).numpy(),
        "E_M": torch.cat(mirror_error).numpy(),
        "C_R": torch.cat(rotation_consistency).numpy(),
        "E_R": torch.cat(rotation_error).numpy(),
    }
    summary = {
        "MSE_h": horizon_values.mean(dim=0).tolist(),
        "E5": float(torch.cat(original_e5).mean()),
        "E2_1x": float(torch.cat(original_e2).mean()),
        "C_M": float(torch.cat(mirror_consistency).mean()),
        "E_M": float(torch.cat(mirror_error).mean()),
        "C_R": float(torch.cat(rotation_consistency).mean()),
        "E_R": float(torch.cat(rotation_error).mean()),
        "E2_2x": speed_metrics["2"]["E2"],
        "E5_2x": speed_metrics["2"]["E5"],
        "E2_3x": speed_metrics["3"]["E2"],
        "inference_seconds": inference_seconds,
    }
    return summary, per_video


def load_trained(model_name, seed, checkpoint_dir, device):
    path = Path(checkpoint_dir) / f"model_{model_name}" / f"seed_{seed}" / "final.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    saved = torch.load(path, map_location=device, weights_only=False)
    if int(saved["epoch"]) != 29:
        raise ValueError(f"Checkpoint is not from final epoch: {path}")
    model = build_model(model_name).to(device)
    model.load_state_dict(saved["state_dict"])
    return model, saved


def save_evaluation(artifact_dir, name, seed, summary, per_video):
    metric_dir = Path(artifact_dir) / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    (metric_dir / f"{name}_seed_{seed}.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(metric_dir / f"{name}_seed_{seed}_per_video.npz", **per_video)


def evaluate_learned(model_name, seed, loaders, args, device, symmetrized=False):
    base, saved = load_trained(model_name, seed, args.checkpoint_dir, device)
    predictor = SymmetrizedForecaster(base).to(device) if symmetrized else base
    name = f"sym_{model_name}" if symmetrized else model_name
    summary, per_video = evaluate_predictor(predictor, loaders, device, f"{name} seed {seed}")
    summary.update(
        {
            "model": name,
            "base_model": model_name,
            "seed": seed,
            "parameters": parameter_count(base),
            "reach": one_step_reach(model_name),
            "training_seconds_per_epoch": float(np.mean([item["training_seconds"] for item in saved["history"]])),
            "hardware": saved["hardware"],
            "amp_training": bool(saved["amp"]),
        }
    )
    save_evaluation(args.artifact_dir, name, seed, summary, per_video)
    print(f"[{name} | seed {seed}] {json.dumps(summary)}", flush=True)


def evaluate_baselines(loaders, args, device):
    records = {}
    for name, predictor in (("Blank", BlankPredictor()), ("Copy-last", CopyLastPredictor())):
        summary, per_video = evaluate_predictor(predictor.to(device), loaders, device, name, timed=False)
        summary.update({"model": name, "parameters": 0})
        records[name] = summary
        save_evaluation(args.artifact_dir, name.replace("-", "_"), 0, summary, per_video)
    path = Path(args.artifact_dir) / "metrics" / "baselines.json"
    path.write_text(json.dumps(records, indent=2), encoding="utf-8")


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    splits = load_splits(args.data_dir, download=args.download)
    loaders = {speed: make_loader(splits["test"], speed, args.workers) for speed in (1, 2, 3)}
    if args.baselines and args.seed == 0:
        evaluate_baselines(loaders, args, device)
    for model_name in args.models:
        evaluate_learned(model_name, args.seed, loaders, args, device, symmetrized=False)
    for model_name in args.sym_models:
        evaluate_learned(model_name, args.seed, loaders, args, device, symmetrized=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=[])
    parser.add_argument("--sym-models", nargs="*", default=[])
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    args.models = [name.upper() for name in args.models]
    args.sym_models = [name.upper() for name in args.sym_models]
    return args


if __name__ == "__main__":
    run(parse_args())

