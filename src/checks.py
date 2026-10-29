import argparse
import csv
import json
from pathlib import Path

import torch

from .data import frame_numbers, make_window
from .models import validate_architectures
from .transforms import mirror, rotate


def bright_pixel(x):
    location = torch.nonzero(x == 1, as_tuple=False)
    if len(location) != 1:
        raise AssertionError(location)
    frame, _, row, column = location[0].tolist()
    return frame, row, column


def run_checks():
    torch.manual_seed(0)
    x = torch.rand(20, 1, 64, 64)
    p = torch.zeros(20, 1, 64, 64)
    p[7, 0, 10, 20] = 1
    double_mirror = mirror(mirror(x))
    four_rotations = rotate(x, 4)
    mrm = mirror(rotate(mirror(x), 1))
    inverse_rotation = rotate(x, -1)
    frame_video = torch.arange(1, 21, dtype=torch.float32).reshape(20, 1, 1, 1).expand(20, 1, 64, 64)
    rows = [
        {
            "check": "(a) max|M(M(x))-x|",
            "expected": "0 and torch.equal=True",
            "yours": f"{float((double_mirror - x).abs().max()):g} and torch.equal={torch.equal(double_mirror, x)}",
            "passed": torch.equal(double_mirror, x),
        },
        {
            "check": "(b) max|R4(x)-x|",
            "expected": "0 and torch.equal=True",
            "yours": f"{float((four_rotations - x).abs().max()):g} and torch.equal={torch.equal(four_rotations, x)}",
            "passed": torch.equal(four_rotations, x),
        },
        {
            "check": "(c) max|MRM(x)-R^-1(x)|",
            "expected": "0 and torch.equal=True",
            "yours": f"{float((mrm - inverse_rotation).abs().max()):g} and torch.equal={torch.equal(mrm, inverse_rotation)}",
            "passed": torch.equal(mrm, inverse_rotation),
        },
        {
            "check": "(d) bright pixel of M(p)",
            "expected": "(7, 10, 43)",
            "yours": str(bright_pixel(mirror(p))),
            "passed": bright_pixel(mirror(p)) == (7, 10, 43),
        },
        {
            "check": "(e) bright pixel of R(p)",
            "expected": "(7, 43, 10)",
            "yours": str(bright_pixel(rotate(p, 1))),
            "passed": bright_pixel(rotate(p, 1)) == (7, 43, 10),
        },
    ]
    expected_frames = {
        1: ([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]),
        2: ([1, 3, 5, 7, 9], [11, 13, 15, 17, 19]),
        3: ([1, 4, 7, 10, 13], [16, 19]),
    }
    for label, speed in zip(("(f)", "(g)", "(h)"), (1, 2, 3)):
        inputs, targets, mask = make_window(frame_video, speed=speed, start=1)
        input_values = inputs[:, 0, 0, 0].to(torch.int64).tolist()
        target_values = targets[mask, 0, 0, 0].to(torch.int64).tolist()
        reported = frame_numbers(speed, 1)
        passed = (input_values, target_values) == expected_frames[speed] and reported == expected_frames[speed]
        rows.append(
            {
                "check": f"{label} {speed}x input / target frames",
                "expected": f"{expected_frames[speed][0]} / {expected_frames[speed][1]}",
                "yours": f"{input_values} / {target_values}",
                "passed": passed,
            }
        )
    if not all(row["passed"] for row in rows):
        raise AssertionError(rows)
    return rows, validate_architectures()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    args = parser.parse_args()
    rows, architectures = run_checks()
    table_dir = args.artifact_dir / "tables"
    metric_dir = args.artifact_dir / "metrics"
    table_dir.mkdir(parents=True, exist_ok=True)
    metric_dir.mkdir(parents=True, exist_ok=True)
    with (table_dir / "table_1.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("check", "expected", "yours"))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in writer.fieldnames})
    (metric_dir / "checks.json").write_text(
        json.dumps({"table_1": rows, "architectures": architectures}, indent=2),
        encoding="utf-8",
    )
    for row in rows:
        print(f"{row['check']}: {row['yours']}", flush=True)
    print(architectures, flush=True)


if __name__ == "__main__":
    main()

