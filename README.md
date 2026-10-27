# WavePrism

**Global probabilistic ocean-wave forecasting with physically diverse experts and satellite-observation constraints.**

WavePrism learns from multiple physically diverse wave-model experts and observational constraints to produce global probabilistic significant-wave-height predictions. This repository organizes the model, training stages, data readers, and inference workflows as a Python package.

## Repository structure

```text
WavePrism/
├── README.md
├── requirements.txt
├── pyproject.toml
├── .gitignore
└── waveprism/
    ├── models/
    │   ├── expert_forecaster.py
    │   ├── expert_selector.py
    │   ├── probabilistic_forecaster.py
    │   └── factory.py
    ├── training/
    │   ├── train_experts.py
    │   ├── train_selector.py
    │   ├── train_probability.py
    │   └── common.py
    ├── data/
    │   ├── hdf5.py
    │   ├── era5.py
    │   ├── ifs_wind.py
    │   └── forcing.py
    └── inference/
        ├── common.py
        ├── era5_backtest.py
        ├── ifs_control.py
        └── ifs_ensemble.py
```

## 1. Environment setup

Use Python 3.10 or later, PyTorch 2.4 or later, and NumPy 2.x. Install PyTorch separately to match the CUDA version available on your computer, using the [official PyTorch installation selector](https://pytorch.org/get-started/locally/). Then, from the repository root:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Or on Linux:

```bash
source .venv/bin/activate
```

Install the project and its declared dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` installs the project in editable mode. Runtime dependencies are declared in `pyproject.toml`. Install a CUDA-enabled PyTorch build before running GPU training or inference.

Verify the environment:

```bash
python -c "import torch, h5py, netCDF4, numpy; print('torch', torch.__version__, 'CUDA available:', torch.cuda.is_available())"
```

## 2. Local case directory for inference

The inference commands below are written for the following case layout. Place the `case` directory at the repository root, next to `README.md` and `waveprism/`.

```text
case/
├── analysis/                         # Historical analysis winds for IFS forecasts
│   ├── 10m_wind_20251102_0000.nc
│   ├── 10m_wind_20251102_0600.nc
│   └── ...                            # One file every 6 hours
├── cf/                               # IFS control-member forecast winds
│   ├── ifs_control_u10v10_20251112_00.nc
│   ├── ifs_control_u10v10_20251113_00.nc
│   └── ...
├── pf/                               # IFS perturbed/ensemble forecast winds
│   ├── ifs_ens_u10v10_20251112_00.nc
│   ├── ifs_ens_u10v10_20251113_00.nc
│   └── ...
├── model_static/                     # Model checkpoints and static geographic input
│   ├── model_stage1.pt
│   ├── model_stage2.pt
│   ├── model_stage3.pt
│   └── static.npz
├── era5/                             # Hourly ERA5 u10/v10 files for backtesting
│   ├── ERA5_u10v10_20250101_0000.nc
│   ├── ERA5_u10v10_20250101_0100.nc
│   └── ...
└── output/                           # Forecast outputs; created automatically as needed
```

## 3. Main program usage

### 3.1 Stage 1 — train the expert forecaster

This stage trains the wind-history encoder and the 21 independent expert decoders. Replace the training-data paths with the actual locations of your HDF5 dataset and static features. The `case/` folder above is the inference case; it is not the training dataset.

```bash
python -m waveprism.training.train_experts train \
  --data-h5 PATH/TO/global_wave_dataset_base_1h_0p5deg.h5 \
  --static-npz PATH/TO/static_etopo5_3features_global_h5grid_0p5deg.npz \
  --output outputs/stage1 \
  --epochs 200 --batch-size 1 --workers 0
```

Optional Stage-1 utilities:

```bash
python -m waveprism.training.train_experts export --help
```

### 3.2 Stage 2 — train the expert selector

Stage 2 uses the Stage-1 checkpoint and satellite-observation supervision to train the selector.

```bash
python -m waveprism.training.train_selector \
  --stage1-checkpoint PATH/TO/stage1_best.pt \
  --data-h5 PATH/TO/global_wave_dataset_base_1h_0p5deg.h5 \
  --static-npz PATH/TO/static_etopo5_3features_global_h5grid_0p5deg.npz \
  --save-dir outputs/stage2 \
  --batch-size 2 --workers 0 --epochs 50 --encoder-amp
```

### 3.3 Stage 3 — train the probabilistic predictor

Stage 3 freezes the compatible Stage-1 and Stage-2 components and trains the probabilistic residual module.

```bash
python -m waveprism.training.train_probability \
  --stage1-checkpoint PATH/TO/stage1_best.pt \
  --selector-checkpoint PATH/TO/stage2_selector_best.pt \
  --data-h5 PATH/TO/global_wave_dataset_base_1h_0p5deg.h5 \
  --static-npz PATH/TO/static_etopo5_3features_global_h5grid_0p5deg.npz \
  --save-dir outputs/stage3 \
  --batch-size 2 --workers 0 --epochs 50 --lr 1e-4
```

After training, place the compatible Stage-1, Stage-2, and Stage-3 checkpoints in `case/model_static/` using the names shown in the case layout, or edit the checkpoint arguments in the inference commands below. Do not mix checkpoints from different training runs.

### 3.4 ERA5 — hourly backtesting

This program reads `ERA5_u10v10_YYYYMMDD_HHMM.nc` files from `case/era5/`, constructs hourly historical wind windows, and writes a NetCDF forecast/backtest file. `--year` selects target times in that calendar year; the program skips target times without a complete continuous input history unless `--invalid-history error` is selected.

```bash
python -m waveprism.inference.era5_backtest \
  --era5-dir case/era5 \
  --static-npz case/model_static/static.npz \
  --stage1-checkpoint case/model_static/model_stage1.pt \
  --stage2-checkpoint case/model_static/model_stage2.pt \
  --stage3-checkpoint case/model_static/model_stage3.pt \
  --year 2025 \
  --output-nc case/output/era5_backtest_2025.nc \
  --batch-size 1 --workers 0 --device cuda:0 --resume
```

For a selected UTC interval, use an inclusive `--start` and exclusive `--end` instead of `--year`. For example, to request 11 January 2025 only:

```bash
python -m waveprism.inference.era5_backtest \
  --era5-dir case/era5 \
  --static-npz case/model_static/static.npz \
  --stage1-checkpoint case/model_static/model_stage1.pt \
  --stage2-checkpoint case/model_static/model_stage2.pt \
  --stage3-checkpoint case/model_static/model_stage3.pt \
  --start 2025-01-11T00:00:00 --end 2025-01-12T00:00:00 \
  --output-nc case/output/era5_backtest_20250111.nc \
  --batch-size 1 --workers 0 --device cuda:0 --resume
```

The selected targets must have the complete history required by the Stage-1 checkpoint (for the standard 241-sample model, 240 hours before each target through the target hour). If your case does not contain that history, the earliest target times will be skipped.

### 3.5 IFS — control-member forecast

This program reads historical analysis winds from `case/analysis/` and control forecast winds from `case/cf/`. It selects the latest available 00 UTC initialization with complete historical analysis by default when `--latest` is used. The model produces forecast steps 0, 6, ..., 240 hours. IFS u10/v10 anchors at 6-hour intervals are linearly interpolated to the hourly model input; there is no spatial interpolation.

First validate the inputs without running the model:

```bash
python -m waveprism.inference.ifs_control \
  --analysis-dir case/analysis \
  --forecast-dir case/cf \
  --static-npz case/model_static/static.npz \
  --stage1-checkpoint case/model_static/model_stage1.pt \
  --stage2-checkpoint case/model_static/model_stage2.pt \
  --stage3-checkpoint case/model_static/model_stage3.pt \
  --latest --check-only
```

Then run the forecast:

```bash
python -m waveprism.inference.ifs_control \
  --analysis-dir case/analysis \
  --forecast-dir case/cf \
  --static-npz case/model_static/static.npz \
  --stage1-checkpoint case/model_static/model_stage1.pt \
  --stage2-checkpoint case/model_static/model_stage2.pt \
  --stage3-checkpoint case/model_static/model_stage3.pt \
  --latest \
  --output-dir case/output/ifs_control \
  --device cuda:0 --resume
```

To process a fixed inclusive initialization-date range instead of the latest date, replace `--latest` with `--start-date 20251112 --end-date 20251113`. Both dates are included. `--t0-source forecast` is the default and uses the forecast step-0 wind; `--t0-source analysis` requires the corresponding analysis wind at the initialization time.

### 3.6 IFS — perturbed ensemble probabilistic forecast

This program reads the same historical analysis winds from `case/analysis/`, but takes the ensemble forecast files from `case/pf/`. By default, it processes all members found in the file and requires that the selected members include control member `0`. It writes the final mean and standard deviation into one NetCDF file per initialization, plus a run-summary CSV.

Validate inputs first:

```bash
python -m waveprism.inference.ifs_ensemble \
  --analysis-dir case/analysis \
  --forecast-dir case/pf \
  --static-npz case/model_static/static.npz \
  --stage1-checkpoint case/model_static/model_stage1.pt \
  --stage2-checkpoint case/model_static/model_stage2.pt \
  --stage3-checkpoint case/model_static/model_stage3.pt \
  --latest --control-member 0 --check-only
```

Run the ensemble probabilistic forecast:

```bash
python -m waveprism.inference.ifs_ensemble \
  --analysis-dir case/analysis \
  --forecast-dir case/pf \
  --static-npz case/model_static/static.npz \
  --stage1-checkpoint case/model_static/model_stage1.pt \
  --stage2-checkpoint case/model_static/model_stage2.pt \
  --stage3-checkpoint case/model_static/model_stage3.pt \
  --latest --control-member 0 \
  --output-dir case/output/ifs_ensemble \
  --device cuda:0 --resume
```

For a fixed inclusive date range, replace `--latest` with `--start-date 20251112 --end-date 20251113`. To run a specific subset of members, add `--member-ids 0 1 2 3`; omit that option to use all members in the input NetCDF. The ensemble file must contain a member-coordinate dimension and control member ID `0`. A perturbed-only file containing members 1–50 is insufficient for this workflow; prepare an ensemble input containing member 0 as well. The current ensemble reader does not combine `cf/` and `pf/` automatically.

### 3.7 Selecting stage outputs

ERA5 backtesting and IFS control forecasting save only Stage-3 results by default. Use `--stages` to select the outputs to save:

| Option | Saved output |
| --- | --- |
| `--stages 1` | Wave heights from the Stage-1 expert decoders |
| `--stages 2` | Selected expert wave height, expert index, selection confidence, and expert disagreement |
| `--stages 3` | Final predictive mean and standard deviation |
| `--stages 1 2 3` | Outputs from all three stages |
| `--stages 1 3` | Stage-1 expert predictions and Stage-3 mean and standard deviation |

Add the selected option to the ERA5 or IFS control command above. For example, add `--stages 1` to save only expert predictions, or `--stages 1 2 3` to save all stage outputs.

| Highest requested stage | Required checkpoint arguments |
| --- | --- |
| Stage 1 | `--stage1-checkpoint` |
| Stage 2 | `--stage1-checkpoint` and `--stage2-checkpoint` |
| Stage 3 | `--stage1-checkpoint` and `--stage3-checkpoint` |

Stage-3 checkpoints include the trained selector; `--stage2-checkpoint` is optional when requesting Stage 3. Later stages still run their upstream components, even when only the final results are saved.

For Stage-1 output, use `--expert-ids 1 3 5` to save selected experts or `--save-z` to also save latent features. Expert IDs start at 1. For Stage-2 output, add `--save-selection-probabilities` to save probabilities for all experts.

The IFS ensemble program outputs only the final Stage-3 mean and standard deviation and does not use the `--stages` option.

## 4. Output files

By default, the output directories are created automatically:

- ERA5 backtest: `case/output/era5_backtest_2025.nc` (or the requested output filename).
- IFS control: one file per initialization, such as `waveprism_20251112_00.nc`, under `case/output/ifs_control/`.
- IFS ensemble: one `waveprism_ensemble_YYYYMMDD_00.nc` per initialization, containing the final mean and standard deviation, plus a run-summary CSV under `case/output/ifs_ensemble/`.
