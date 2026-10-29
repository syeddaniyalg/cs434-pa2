import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch

from .data import TEST_RANGE, load_splits, make_window
from .models import build_model
from .symmetrize import SymmetrizedForecaster
from .transforms import apply_group


def load_model(model_name, seed, checkpoint_dir, device, symmetrized=False):
    path = Path(checkpoint_dir) / f"model_{model_name}" / f"seed_{seed}" / "final.pt"
    saved = torch.load(path, map_location=device, weights_only=False)
    model = build_model(model_name).to(device)
    model.load_state_dict(saved["state_dict"])
    model.eval()
    return SymmetrizedForecaster(model).to(device).eval() if symmetrized else model


def predict(model, inputs, device):
    with torch.no_grad():
        return model(inputs.unsqueeze(0).to(device)).squeeze(0).float().cpu()


def save_figure(fig, figure_dir, name):
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / f"{name}.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def show_frame(ax, frame, title=None):
    ax.imshow(frame.squeeze().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=8)


def component_count(frame):
    binary = (frame.squeeze().numpy() > 0.2).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= 20 for index in range(1, count))


def has_overlap(video, frame_indices):
    return any(component_count(video[index]) <= 1 for index in frame_indices)


def select_figure2_ids(test_videos):
    overlap = None
    separate = []
    for index in range(len(test_videos)):
        video, video_id = test_videos[index]
        overlapping = has_overlap(video, range(10))
        if overlapping and overlap is None:
            overlap = video_id
        if not overlapping and len(separate) < 2:
            separate.append(video_id)
        if overlap is not None and len(separate) == 2:
            return [overlap, *separate]
    raise RuntimeError("Could not select Figure 2 videos")


def figure5_displayed_frames():
    return tuple(sorted({
        frame
        for speed in (1, 2, 3)
        for frame in (torch.arange(7) * speed).tolist()
    }))


def select_figure5_id(test_videos):
    displayed_frames = figure5_displayed_frames()
    for index in range(len(test_videos)):
        video, video_id = test_videos[index]
        if not has_overlap(video, displayed_frames):
            return video_id
    raise RuntimeError("Could not select a non-overlapping Figure 5 video")


def video_by_id(test_videos, video_id):
    video, actual_id = test_videos[int(video_id) - TEST_RANGE[0]]
    if actual_id != int(video_id):
        raise AssertionError((actual_id, video_id))
    return video


def figure2(model, test_videos, selected, device, figure_dir, metric_dir):
    fig, axes = plt.subplots(6, 10, figsize=(15, 9))
    saved = {}
    for case, video_id in enumerate(selected):
        video = video_by_id(test_videos, video_id)
        inputs, targets, _ = make_window(video, 1, 1)
        predictions = predict(model, inputs, device)
        saved[str(video_id)] = {"inputs": inputs, "targets": targets, "predictions": predictions}
        prediction_row = 2 * case
        truth_row = prediction_row + 1
        for column in range(5):
            show_frame(axes[prediction_row, column], inputs[column], f"x{column + 1}" if case == 0 else None)
            show_frame(axes[truth_row, column], inputs[column])
        for horizon in range(5):
            show_frame(axes[prediction_row, 5 + horizon], predictions[horizon], f"pred {horizon + 1}" if case == 0 else None)
            show_frame(axes[truth_row, 5 + horizon], targets[horizon], f"true {horizon + 1}" if case == 0 else None)
        axes[prediction_row, 0].set_ylabel(f"Video {video_id}\nPrediction", fontsize=8)
        axes[truth_row, 0].set_ylabel("Ground truth", fontsize=8)
    fig.suptitle("Figure 2: Model A forecasts on three test videos")
    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_2")
    torch.save(saved, metric_dir / "figure_2_cases.pt")


def transformation_case(model, video, video_id, group, device):
    inputs, targets, _ = make_window(video, 1, 1)
    original_predictions = predict(model, inputs, device)
    transformed_inputs = apply_group(inputs, group)
    transformed_targets = apply_group(targets, group)
    transformed_predictions = predict(model, transformed_inputs, device)
    transformed_original_predictions = apply_group(original_predictions, group)
    return {
        "video_id": int(video_id),
        "group": int(group),
        "inputs": inputs,
        "targets": targets,
        "original_predictions": original_predictions,
        "transformed_inputs": transformed_inputs,
        "transformed_targets": transformed_targets,
        "transformed_predictions": transformed_predictions,
        "transformed_original_predictions": transformed_original_predictions,
    }


def case_rows(case):
    return [
        torch.cat((case["inputs"], case["original_predictions"]), dim=0),
        torch.cat((case["transformed_inputs"], case["transformed_targets"]), dim=0),
        torch.cat((case["transformed_inputs"], case["transformed_predictions"]), dim=0),
        torch.cat((case["transformed_inputs"], case["transformed_original_predictions"]), dim=0),
    ]


def draw_case(case, title, highlight=None):
    rows = case_rows(case)
    labels = ("X | f(X)", "T(X) | T(Y)", "T(X) | f(T(X))", "T(X) | T(f(X))")
    fig, axes = plt.subplots(4, 10, figsize=(15, 6))
    for row in range(4):
        for column in range(10):
            show_frame(axes[row, column], rows[row][column])
            if highlight is not None and column == 5 + highlight:
                for spine in axes[row, column].spines.values():
                    spine.set_visible(True)
                    spine.set_color("crimson")
                    spine.set_linewidth(2.5)
        axes[row, 0].set_ylabel(labels[row], fontsize=8)
    for column in range(5):
        axes[0, column].set_title(f"input {column + 1}", fontsize=8)
    for column in range(5, 10):
        axes[0, column].set_title(f"h={column - 4}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def figure_to_image(fig):
    fig.canvas.draw()
    array = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(array)


def save_case_gif(case, title, path):
    frames = [figure_to_image(draw_case(case, title, highlight=horizon)) for horizon in range(5)]
    path.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(path, save_all=True, append_images=frames[1:], duration=900, loop=0)


def transformation_figures(model, test_videos, mirror_ids, rotation_ids, device, figure_dir, gif_dir, number):
    cases = []
    for group, label, ids in ((4, "mirror", mirror_ids), (1, "rotation", rotation_ids)):
        for rank, video_id in enumerate(ids, 1):
            video = video_by_id(test_videos, video_id)
            case = transformation_case(model, video, video_id, group, device)
            cases.append((label, rank, case))
            title = f"Figure {number}: {label}, video {video_id}"
            individual = draw_case(case, title)
            save_figure(individual, figure_dir / f"figure_{number}_panels", f"{label}_video_{video_id}")
            save_case_gif(case, title, gif_dir / f"figure_{number}_{label}_video_{video_id}.gif")
    fig, axes = plt.subplots(16, 10, figsize=(15, 23))
    labels = ("X | f(X)", "T(X) | T(Y)", "T(X) | f(T(X))", "T(X) | T(f(X))")
    for block, (label, rank, case) in enumerate(cases):
        rows = case_rows(case)
        for row in range(4):
            axis_row = block * 4 + row
            for column in range(10):
                show_frame(axes[axis_row, column], rows[row][column])
            axes[axis_row, 0].set_ylabel(labels[row], fontsize=7)
        axes[block * 4, 0].text(
            -0.65,
            1.25,
            f"{label.title()} rank {rank}, video {case['video_id']}",
            transform=axes[block * 4, 0].transAxes,
            fontsize=10,
            fontweight="bold",
        )
    fig.suptitle(f"Figure {number}: transformation consistency cases")
    fig.tight_layout()
    composite_name = f"figure_{number}a" if number == 6 else f"figure_{number}"
    save_figure(fig, figure_dir, composite_name)
    return cases


def figure4(per_video, mirror_ids, rotation_ids, figure_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for axis, key, ids, title in (
        (axes[0], "C_M", mirror_ids, "Mirror consistency gap"),
        (axes[1], "C_R", rotation_ids, "Rotation consistency gap"),
    ):
        axis.hist(per_video[key], bins=30, edgecolor="black")
        for video_id in ids:
            location = int(np.where(per_video["video_index"] == video_id)[0][0])
            axis.axvline(per_video[key][location], linestyle="--", label=f"video {video_id}")
        axis.set(xlabel=key, ylabel="Test videos", title=title)
        axis.legend()
    fig.suptitle("Figure 4: per-video consistency gaps for seed-0 model A")
    fig.tight_layout()
    save_figure(fig, figure_dir, "figure_4")


def speed_case(model, video, device):
    records = {}
    for speed in (1, 2, 3):
        inputs, targets, mask = make_window(video, speed, 1)
        records[speed] = {
            "inputs": inputs,
            "targets": targets,
            "mask": mask,
            "predictions": predict(model, inputs, device),
        }
    return records


def draw_speed_figure(records, video_id, number, title_suffix):
    fig, axes = plt.subplots(6, 7, figsize=(11, 9))
    for block, speed in enumerate((1, 2, 3)):
        prediction_row = block * 2
        truth_row = prediction_row + 1
        record = records[speed]
        for column in range(5):
            show_frame(axes[prediction_row, column], record["inputs"][column])
            show_frame(axes[truth_row, column], record["inputs"][column])
        for horizon in range(2):
            show_frame(axes[prediction_row, 5 + horizon], record["predictions"][horizon])
            show_frame(axes[truth_row, 5 + horizon], record["targets"][horizon])
        axes[prediction_row, 0].set_ylabel(f"{speed}x prediction", fontsize=8)
        axes[truth_row, 0].set_ylabel(f"{speed}x truth", fontsize=8)
    for column in range(5):
        axes[0, column].set_title(f"input {column + 1}", fontsize=8)
    axes[0, 5].set_title("h=1", fontsize=8)
    axes[0, 6].set_title("h=2", fontsize=8)
    fig.suptitle(f"Figure {number}: video {video_id}, {title_suffix}")
    fig.tight_layout()
    return fig


def baseline_stage(args, test_videos, figure_dir, gif_dir, metric_dir, device):
    model = load_model("A", 0, args.checkpoint_dir, device)
    per_video = np.load(metric_dir / "A_seed_0_per_video.npz")
    mirror_order = np.argsort(-per_video["C_M"])[:2]
    rotation_order = np.argsort(-per_video["C_R"])[:2]
    mirror_ids = per_video["video_index"][mirror_order].astype(int).tolist()
    rotation_ids = per_video["video_index"][rotation_order].astype(int).tolist()
    figure2_ids = select_figure2_ids(test_videos)
    figure5_id = select_figure5_id(test_videos)
    selected = {
        "figure_2": figure2_ids,
        "figure_2_overlap_video": figure2_ids[0],
        "figure_3_mirror": mirror_ids,
        "figure_3_rotation": rotation_ids,
        "figure_5": figure5_id,
        "figure_5_displayed_true_frame_indices": list(figure5_displayed_frames()),
        "figure_5_displayed_true_component_counts": [
            component_count(video_by_id(test_videos, figure5_id)[index])
            for index in figure5_displayed_frames()
        ],
    }
    (metric_dir / "selected_cases.json").write_text(json.dumps(selected, indent=2), encoding="utf-8")
    figure2(model, test_videos, figure2_ids, device, figure_dir, metric_dir)
    cases = transformation_figures(model, test_videos, mirror_ids, rotation_ids, device, figure_dir, gif_dir, 3)
    torch.save(cases, metric_dir / "figure_3_cases.pt")
    figure4(per_video, mirror_ids, rotation_ids, figure_dir)
    video = video_by_id(test_videos, figure5_id)
    records = speed_case(model, video, device)
    save_figure(draw_speed_figure(records, figure5_id, 5, "seed-0 model A"), figure_dir, "figure_5")
    torch.save(records, metric_dir / "figure_5_case.pt")


def final_stage(args, test_videos, figure_dir, gif_dir, metric_dir, device):
    selected = json.loads((metric_dir / "selected_cases.json").read_text(encoding="utf-8"))
    model = load_model(args.final_base, 0, args.checkpoint_dir, device, symmetrized=True)
    cases = transformation_figures(
        model,
        test_videos,
        selected["figure_3_mirror"],
        selected["figure_3_rotation"],
        device,
        figure_dir,
        gif_dir,
        6,
    )
    video = video_by_id(test_videos, selected["figure_5"])
    records = speed_case(model, video, device)
    speed_figure = draw_speed_figure(records, selected["figure_5"], 6, f"final symmetrized {args.final_base}")
    save_figure(speed_figure, figure_dir, "figure_6b")
    torch.save({"transformation_cases": cases, "speed_case": records}, metric_dir / "figure_6_cases.pt")


def speed_stage(args, test_videos, figure_dir, metric_dir, device):
    selected_path = metric_dir / "selected_cases.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    video_id = select_figure5_id(test_videos)
    displayed_frames = figure5_displayed_frames()
    video = video_by_id(test_videos, video_id)
    selected["figure_5"] = video_id
    selected["figure_5_displayed_true_frame_indices"] = list(displayed_frames)
    selected["figure_5_displayed_true_component_counts"] = [
        component_count(video[index]) for index in displayed_frames
    ]
    selected_path.write_text(json.dumps(selected, indent=2), encoding="utf-8")

    model_a = load_model("A", 0, args.checkpoint_dir, device)
    model_a_records = speed_case(model_a, video, device)
    save_figure(draw_speed_figure(model_a_records, video_id, 5, "seed-0 model A"), figure_dir, "figure_5")
    torch.save(model_a_records, metric_dir / "figure_5_case.pt")

    final_model = load_model(args.final_base, 0, args.checkpoint_dir, device, symmetrized=True)
    final_records = speed_case(final_model, video, device)
    final_figure = draw_speed_figure(final_records, video_id, 6, f"final symmetrized {args.final_base}")
    save_figure(final_figure, figure_dir, "figure_6b")
    figure_6_path = metric_dir / "figure_6_cases.pt"
    figure_6_cases = torch.load(figure_6_path, map_location="cpu", weights_only=False)
    figure_6_cases["speed_case"] = final_records
    torch.save(figure_6_cases, figure_6_path)


def run(args):
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    artifact_dir = Path(args.artifact_dir)
    figure_dir = artifact_dir / "figures"
    gif_dir = artifact_dir / "gifs"
    metric_dir = artifact_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    splits = load_splits(args.data_dir, download=args.download)
    if args.stage in {"baseline", "all"}:
        baseline_stage(args, splits["test"], figure_dir, gif_dir, metric_dir, device)
    if args.stage in {"final", "all"}:
        final_stage(args, splits["test"], figure_dir, gif_dir, metric_dir, device)
    if args.stage == "speed":
        speed_stage(args, splits["test"], figure_dir, metric_dir, device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("baseline", "final", "speed", "all"), required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--final-base", choices=("D", "DB"), default="D")
    parser.add_argument("--download", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())

