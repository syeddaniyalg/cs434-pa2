import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import EpochSampler, FixedWindows, TrainingWindows, load_splits
from .metrics import masked_per_video_error, masked_training_loss
from .models import build_model, one_step_reach, parameter_count, validate_architectures


BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def loader_options(workers):
    options = {
        "num_workers": workers,
        "pin_memory": True,
        "persistent_workers": workers > 0,
    }
    if workers > 0:
        options["prefetch_factor"] = 4
    return options


def move_batch(batch, device):
    inputs = batch["inputs"].to(device, non_blocking=True)
    targets = batch["targets"].to(device, non_blocking=True)
    mask = batch["mask"].to(device, non_blocking=True)
    return inputs, targets, mask


class RunLogger:
    def __init__(self, model_name, seed, log_dir):
        self.prefix = f"[{model_name} | seed {seed}]"
        self.text_path = log_dir / f"model_{model_name}_seed_{seed}.log"
        self.jsonl_path = log_dir / f"model_{model_name}_seed_{seed}.jsonl"
        log_dir.mkdir(parents=True, exist_ok=True)

    def write(self, message, record=None):
        line = f"{self.prefix} {message}"
        print(line, flush=True)
        with self.text_path.open("a", encoding="utf-8") as stream:
            stream.write(line + "\n")
        if record is not None:
            with self.jsonl_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record) + "\n")


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_validation(model, loader, device, description, use_amp):
    model.eval()
    losses = []
    errors = []
    with torch.no_grad():
        progress = tqdm(loader, desc=description, dynamic_ncols=True, leave=True, file=sys.stdout)
        for batch in progress:
            inputs, targets, mask = move_batch(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions = model(inputs)
                loss = masked_training_loss(predictions, targets, mask)
            losses.append(loss.detach().cpu())
            errors.append(masked_per_video_error(predictions.float(), targets, mask).detach().cpu())
            progress.set_postfix(loss=f"{loss.item():.6f}")
    return float(torch.stack(losses).mean()), float(torch.cat(errors).mean())


def checkpoint_payload(model, optimizer, scaler, model_name, seed, epoch, history, device, use_amp):
    return {
        "model_name": model_name,
        "seed": seed,
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "history": history,
        "parameters": parameter_count(model),
        "reach": one_step_reach(model_name),
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "amp": use_amp,
        "device": str(device),
        "hardware": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_version": torch.__version__,
        "numpy_version": np.__version__,
    }


def train(args):
    model_name = args.model.upper()
    seed_everything(args.seed)
    validate_architectures()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
    use_amp = bool(args.amp and device.type == "cuda")
    artifact_dir = Path(args.artifact_dir)
    checkpoint_dir = Path(args.checkpoint_dir) / f"model_{model_name}" / f"seed_{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(model_name, args.seed, artifact_dir / "logs")
    splits = load_splits(args.data_dir, download=args.download)
    train_dataset = TrainingWindows(splits["train"], model_name, args.seed)
    sampler = EpochSampler(train_dataset, args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        drop_last=False,
        **loader_options(args.workers),
    )
    validation_loader = DataLoader(
        FixedWindows(splits["validation"], speed=1, start=1),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        **loader_options(args.workers),
    )
    model = build_model(model_name).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    history = []
    start_epoch = 0
    latest_path = checkpoint_dir / "latest.pt"
    final_path = checkpoint_dir / "final.pt"
    if args.resume and latest_path.exists():
        saved = torch.load(latest_path, map_location=device, weights_only=False)
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        history = saved["history"]
        start_epoch = int(saved["epoch"]) + 1
        logger.write(f"Resuming from epoch {start_epoch + 1}")
    logger.write(
        f"device={device} parameters={parameter_count(model)} reach={one_step_reach(model_name)} amp={use_amp}"
    )
    for epoch in range(start_epoch, EPOCHS):
        sampler.set_epoch(epoch)
        model.train()
        running_loss = 0.0
        seen = 0
        synchronize(device)
        epoch_start = time.perf_counter()
        progress = tqdm(
            train_loader,
            desc=f"{model_name} seed {args.seed} train {epoch + 1:02d}/{EPOCHS}",
            dynamic_ncols=True,
            leave=True,
            file=sys.stdout,
            mininterval=1.0,
        )
        for batch in progress:
            inputs, targets, mask = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                predictions = model(inputs)
                loss = masked_training_loss(predictions, targets, mask)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = inputs.shape[0]
            running_loss += loss.detach().item() * batch_size
            seen += batch_size
            progress.set_postfix(loss=f"{running_loss / seen:.6f}")
        synchronize(device)
        training_seconds = time.perf_counter() - epoch_start
        training_loss = running_loss / seen
        validation_start = time.perf_counter()
        validation_loss, validation_e5 = evaluate_validation(
            model,
            validation_loader,
            device,
            f"{model_name} seed {args.seed} validation {epoch + 1:02d}/{EPOCHS}",
            use_amp,
        )
        validation_seconds = time.perf_counter() - validation_start
        record = {
            "epoch": epoch + 1,
            "training_loss": training_loss,
            "validation_loss": validation_loss,
            "validation_e5": validation_e5,
            "training_seconds": training_seconds,
            "validation_seconds": validation_seconds,
        }
        history.append(record)
        payload = checkpoint_payload(
            model,
            optimizer,
            scaler,
            model_name,
            args.seed,
            epoch,
            history,
            device,
            use_amp,
        )
        torch.save(payload, latest_path)
        logger.write(
            f"epoch={epoch + 1:02d}/{EPOCHS} train_loss={training_loss:.6f} "
            f"val_loss={validation_loss:.6f} val_E5={validation_e5:.6f} "
            f"train_seconds={training_seconds:.2f} val_seconds={validation_seconds:.2f}",
            record,
        )
    if len(history) != EPOCHS:
        raise RuntimeError(f"Expected {EPOCHS} completed epochs, found {len(history)}")
    final_payload = checkpoint_payload(
        model,
        optimizer,
        scaler,
        model_name,
        args.seed,
        EPOCHS - 1,
        history,
        device,
        use_amp,
    )
    torch.save(final_payload, final_path)
    history_path = artifact_dir / "metrics" / f"model_{model_name}_seed_{args.seed}_training.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(final_payload | {"state_dict": None, "optimizer": None, "scaler": None}, indent=2), encoding="utf-8")
    mean_seconds = float(np.mean([entry["training_seconds"] for entry in history]))
    logger.write(f"completed final_epoch={EPOCHS} mean_training_seconds={mean_seconds:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=("A", "B", "C", "D", "DB"))
    parser.add_argument("--seed", required=True, type=int, choices=(0, 1))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())

