import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def metric_files(artifact_dir, name):
    metric_dir = Path(artifact_dir) / "metrics"
    return [read_json(metric_dir / f"{name}_seed_{seed}.json") for seed in (0, 1)]


def three_significant(value, scientific_zero=False):
    value = float(value)
    if value == 0:
        return "0.00e+00" if scientific_zero else "0"
    if abs(value) < 1e-3:
        return f"{value:.2e}"
    return f"{value:.3g}"


def seed_entry(records, key, scientific_zero=False):
    values = [float(record[key]) for record in records]
    mean = float(np.mean(values))
    return f"{three_significant(mean, scientific_zero)} ({three_significant(values[0], scientific_zero)}, {three_significant(values[1], scientific_zero)})"


def horizon_entry(records, index):
    values = [float(record["MSE_h"][index]) for record in records]
    return f"{three_significant(np.mean(values))} ({three_significant(values[0])}, {three_significant(values[1])})"


def write_table(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def learned_row(name, records, include_horizons=False, scientific_consistency=False):
    row = {"Model": name}
    if include_horizons:
        for index in range(5):
            row[f"MSE{index + 1}"] = horizon_entry(records, index)
    for key in ("E5", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x"):
        row[key] = seed_entry(records, key, scientific_zero=scientific_consistency and key in {"C_M", "C_R"})
    row["Parameters"] = f"{int(records[0]['parameters']):,}"
    row["Training time / epoch (s)"] = seed_entry(records, "training_seconds_per_epoch")
    row["Inference time (s)"] = seed_entry(records, "inference_seconds")
    return row


def baseline_row(name, record, include_horizons=False):
    row = {"Model": name}
    if include_horizons:
        for index, value in enumerate(record["MSE_h"], 1):
            row[f"MSE{index}"] = three_significant(value)
    for key in ("E5", "C_M", "E_M", "C_R", "E_R", "E2_1x", "E2_2x", "E2_3x"):
        row[key] = three_significant(record[key])
    row["Parameters"] = "0"
    return row


def generate(args):
    artifact_dir = Path(args.artifact_dir)
    table_dir = artifact_dir / "tables"
    metrics = artifact_dir / "metrics"
    baselines = read_json(metrics / "baselines.json")
    a = metric_files(artifact_dir, "A")
    table2_rows = [
        learned_row("A", a, include_horizons=True),
        baseline_row("Blank", baselines["Blank"], include_horizons=True),
        baseline_row("Copy-last", baselines["Copy-last"], include_horizons=True),
    ]
    table2_fields = ["Model", "MSE1", "MSE2", "MSE3", "MSE4", "MSE5", "E5", "Parameters"]
    write_table(table_dir / "table_2.csv", table2_fields, [{key: row[key] for key in table2_fields} for row in table2_rows])
    table3_fields = ["Model", "E5", "C_M", "E_M", "C_R", "E_R"]
    table3_rows = [
        learned_row("A", a),
        baseline_row("Blank", baselines["Blank"]),
        baseline_row("Copy-last", baselines["Copy-last"]),
    ]
    write_table(table_dir / "table_3.csv", table3_fields, [{key: row[key] for key in table3_fields} for row in table3_rows])
    table4_fields = ["Model", "E2_1x", "E2_2x", "E2_3x", "E5_1x", "E5_2x"]
    table4_rows = [
        {
            "Model": "A",
            "E2_1x": seed_entry(a, "E2_1x"),
            "E2_2x": seed_entry(a, "E2_2x"),
            "E2_3x": seed_entry(a, "E2_3x"),
            "E5_1x": seed_entry(a, "E5"),
            "E5_2x": seed_entry(a, "E5_2x"),
        },
        {
            "Model": "Copy-last",
            "E2_1x": three_significant(baselines["Copy-last"]["E2_1x"]),
            "E2_2x": three_significant(baselines["Copy-last"]["E2_2x"]),
            "E2_3x": three_significant(baselines["Copy-last"]["E2_3x"]),
            "E5_1x": three_significant(baselines["Copy-last"]["E5"]),
            "E5_2x": three_significant(baselines["Copy-last"]["E5_2x"]),
        },
    ]
    write_table(table_dir / "table_4.csv", table4_fields, table4_rows)
    if args.stage == "baseline":
        for number in range(2, 5):
            print(table_dir / f"table_{number}.csv", flush=True)
        return
    table5_fields = [
        "Model",
        "E5",
        "C_M",
        "E_M",
        "C_R",
        "E_R",
        "E2_1x",
        "E2_2x",
        "E2_3x",
        "Parameters",
        "Training time / epoch (s)",
        "Inference time (s)",
    ]
    table5_rows = []
    raw = {}
    for name in ("A", "B", "C", "D"):
        records = metric_files(artifact_dir, name)
        raw[name] = records
        row = learned_row(name, records)
        table5_rows.append({key: row[key] for key in table5_fields})
    write_table(table_dir / "table_5.csv", table5_fields, table5_rows)
    table6_rows = []
    for base in ("B", "D"):
        records = metric_files(artifact_dir, f"sym_{base}")
        raw[f"sym_{base}"] = records
        row = learned_row(f"Symmetrized {base}", records, scientific_consistency=True)
        table6_rows.append({key: row[key] for key in table5_fields})
    if args.final_base == "DB":
        records = metric_files(artifact_dir, "sym_DB")
        raw["sym_DB"] = records
        row = learned_row("Final: symmetrized D+B", records, scientific_consistency=True)
        table6_rows.append({key: row[key] for key in table5_fields})
    write_table(table_dir / "table_6.csv", table5_fields, table6_rows)
    (metrics / "aggregate_results.json").write_text(
        json.dumps({"final_base": args.final_base, "records": raw}, indent=2),
        encoding="utf-8",
    )
    for number in range(2, 7):
        print(table_dir / f"table_{number}.csv", flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--final-base", choices=("D", "DB"), default="D")
    parser.add_argument("--stage", choices=("baseline", "full"), default="full")
    return parser.parse_args()


if __name__ == "__main__":
    generate(parse_args())

