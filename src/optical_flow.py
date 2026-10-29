import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from .data import load_splits


def collect_flow(videos):
    directions = []
    speeds = []
    progress = tqdm(range(50), desc="Optical flow 50 training videos", dynamic_ncols=True, leave=True, file=sys.stdout)
    for index in progress:
        video, _ = videos[index]
        for step in range(19):
            previous = np.rint(video[step, 0].numpy() * 255).astype(np.uint8)
            following = np.rint(video[step + 1, 0].numpy() * 255).astype(np.uint8)
            flow = cv2.calcOpticalFlowFarneback(previous, following, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            dx = flow[..., 0]
            dy = flow[..., 1]
            magnitude = np.sqrt(dx * dx + dy * dy)
            keep = (video[step, 0].numpy() > 0.2) & (video[step + 1, 0].numpy() > 0.2) & (magnitude > 0.5)
            directions.append(np.mod(np.degrees(np.arctan2(-dy[keep], dx[keep])), 360.0))
            speeds.append(magnitude[keep])
    return np.concatenate(directions), np.concatenate(speeds)


def plot_flow(directions, speeds, figure_dir):
    direction_edges = np.arange(0, 361, 20)
    direction_counts, _ = np.histogram(directions, bins=direction_edges)
    direction_fraction = direction_counts / direction_counts.sum()
    maximum = max(1.0, float(np.ceil(speeds.max())))
    speed_edges = np.linspace(0, maximum, 31)
    speed_counts, _ = np.histogram(speeds, bins=speed_edges)
    speed_fraction = speed_counts / speed_counts.sum()
    median_speed = float(np.median(speeds))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(direction_edges[:-1], direction_fraction, width=20, align="edge", edgecolor="black")
    axes[0].axhline(1 / 18, color="crimson", linestyle="--", label="Uniform: 1/18")
    axes[0].set(xlabel="Direction (degrees)", ylabel="Fraction of kept pixels", title="(a) Flow direction", xticks=np.arange(0, 361, 60))
    axes[0].legend()
    axes[1].bar(speed_edges[:-1], speed_fraction, width=np.diff(speed_edges), align="edge", edgecolor="black")
    axes[1].axvline(median_speed, color="crimson", linestyle="--", label=f"Median: {median_speed:.3f}")
    axes[1].set(xlabel="Speed (pixels per frame)", ylabel="Fraction of kept pixels", title="(b) Flow speed")
    axes[1].legend()
    fig.suptitle(f"Farneback flow over 950 frame pairs | kept pixels: {len(speeds):,}")
    fig.tight_layout()
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "figure_1.png", dpi=220, bbox_inches="tight")
    fig.savefig(figure_dir / "figure_1.pdf", bbox_inches="tight")
    plt.close(fig)
    return {
        "kept_pixels": int(len(speeds)),
        "median_speed": median_speed,
        "direction_edges": direction_edges.tolist(),
        "direction_fraction": direction_fraction.tolist(),
        "speed_edges": speed_edges.tolist(),
        "speed_fraction": speed_fraction.tolist(),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args()
    splits = load_splits(args.data_dir, download=args.download)
    directions, speeds = collect_flow(splits["train"])
    summary = plot_flow(directions, speeds, args.artifact_dir / "figures")
    metric_dir = args.artifact_dir / "metrics"
    metric_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(metric_dir / "optical_flow.npz", directions=directions, speeds=speeds)
    (metric_dir / "optical_flow.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()

