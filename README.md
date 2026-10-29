# CS434 Programming Assignment 2

This repository contains the implementation and report for CS434 Programming Assignment 2. The work studies spatio-temporal forecasting on Moving MNIST, with particular attention to transformation consistency, unfamiliar motion speeds, and the spatial reach of a ConvLSTM forecaster.

The report also includes the river forecasting proposal from Question 1. The experiments and supporting code in this repository address Question 2.

## Models

All learned models use the same two-layer ConvLSTM forecaster with pixel folding and 480,784 trainable parameters.

| Model | Training data | Convolution dilation | One-step reach |
| --- | --- | ---: | ---: |
| A | Original 1x sequences | 1 | 8 pixels |
| B | 1x sequences with rotation and mirror augmentation | 1 | 8 pixels |
| C | Sequences sampled at 1x, 2x, and 3x | 1 | 8 pixels |
| D | Sequences sampled at 1x, 2x, and 3x | 2 | 16 pixels |

The final predictor applies model D to all eight rotations and reflections, maps the predictions back to the original orientation, and averages them. This gives transformation consistency by construction.

## Repository Structure

```text
cs434-pa2/
|-- notebooks/
|   |-- kaggle_model_a.ipynb
|   `-- kaggle_models_bcd.ipynb
|-- src/
|   |-- checks.py
|   |-- data.py
|   |-- evaluate.py
|   |-- metrics.py
|   |-- models.py
|   |-- optical_flow.py
|   |-- plotting.py
|   |-- results.py
|   |-- symmetrize.py
|   |-- train.py
|   `-- transforms.py
|-- checkpoints/
|-- artifacts/
|   |-- figures/
|   |-- gifs/
|   |-- metrics/
|   `-- tables/
|-- paper/
|   |-- figures/
|   |-- main.tex
|   |-- main.pdf
|   `-- neurips.sty
|-- predictions.md
`-- requirements.txt
```

The main source files have the following roles:

- `data.py` loads Moving MNIST, defines the train, validation, and test splits, and constructs the 1x, 2x, and 3x forecasting windows.
- `models.py` defines the ConvLSTM forecaster and the configurations for models A to D.
- `train.py` trains one model and seed for 30 epochs with mixed precision, progress bars, validation metrics, logs, and resumable checkpoints.
- `evaluate.py` evaluates learned models and simple baselines on forecasting error, transformation consistency, speed, and inference time.
- `symmetrize.py` implements prediction-time averaging over the eight rotations and reflections.
- `checks.py` verifies the transformations, frame selections, parameter count, and model reach.
- `optical_flow.py`, `plotting.py`, and `results.py` generate the figures, GIFs, metrics, and tables used in the report.

## Environment

The experiments use Python 3 with PyTorch, Torchvision, NumPy, Matplotlib, OpenCV, Pillow, and tqdm. The dependencies can be installed with:

```bash
python -m pip install -r requirements.txt
```

Moving MNIST is downloaded through `torchvision.datasets.MovingMNIST` when the download option is enabled. Training was performed on two Tesla T4 GPUs, with one seed assigned to each GPU.

## Kaggle Workflow

The notebooks provide the complete experimental workflow.

1. Run `notebooks/kaggle_model_a.ipynb` with two T4 GPUs and Internet access. It downloads the dataset, runs the exact checks and optical-flow analysis, trains both seeds of model A in parallel, evaluates the baseline, and exports `cs434-pa2-model-a.zip`.
2. Add the exported model A files as a Kaggle dataset. The second notebook expects them under `/kaggle/input/datasets/thesyeddaniyal/cs434-pa2-model-a/`.
3. Run `notebooks/kaggle_models_bcd.ipynb` with two T4 GPUs. It imports the model A results, trains models B, C, and D, evaluates the learned and symmetrized predictors, generates the final tables and figures, and exports the artifact and checkpoint archives.

Both notebooks train seed 0 on `cuda:0` and seed 1 on `cuda:1`. Training and evaluation progress is printed directly in the Kaggle log.

## Command-Line Entry Points

The same stages can be run independently from the repository root. For example:

```bash
python -m src.data --root data --download
python -m src.checks --artifact-dir artifacts
python -m src.optical_flow --data-dir data --artifact-dir artifacts

python -m src.train --model A --seed 0 --device cuda:0 --resume
python -m src.evaluate --models A --seed 0 --device cuda:0

python -m src.results --artifact-dir artifacts --stage full --final-base D
python -m src.plotting --stage final --data-dir data --checkpoint-dir checkpoints --artifact-dir artifacts --device cuda:0 --final-base D
```

Training must be repeated for seeds 0 and 1. Models B, C, and D are selected with the `--model` argument. The evaluation command accepts multiple model names through `--models` and prediction-time symmetrization through `--sym-models`.

## Outputs

- `checkpoints/` contains the final and resumable checkpoints for both seeds of models A to D.
- `artifacts/metrics/` contains training histories, aggregate metrics, consistency measurements, and selected qualitative cases.
- `artifacts/tables/` contains the six CSV tables reported in the paper.
- `artifacts/figures/` contains the quantitative and qualitative figures in PDF and PNG formats.
- `artifacts/gifs/` contains animated transformation cases for models A and D.
- `predictions.md` records the required predictions made before the corresponding experiments.
- `paper/main.pdf` is the complete assignment report.

## Main Findings

Model A performs well on the original 1x task but degrades under faster motion. Speed augmentation reduces the 3x two-horizon error from 0.0459 for model A to 0.0333 for model C. Increasing the one-step reach further reduces it to 0.0318 for model D, although model D has a higher original-sequence error than model A.

Rotation and mirror augmentation reduce average consistency gaps but do not make them exactly zero. Averaging predictions over the eight transformations reduces both consistency gaps to floating-point error. This stronger guarantee increases inference time because the base model is evaluated eight times. The full quantitative results and interpretation are given in `paper/main.pdf`.
