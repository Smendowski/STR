# STR - Selective Temporal Replay

Code and results accompanying the paper **"Bridging Continual Learning and Green Cloud Computing: Foundations for Sustainable Time Series Anomaly Detection"**.

STR is a continual-learning replay strategy for time-series anomaly detection: novelty-gated admission, retention-aware replacement, age-weighted preservation.

## Table of contents
1. [Repository layout](#1-repository-layout)
2. [Environment setup](#2-environment-setup)
3. [Data](#3-data)
4. [Running experiments](#4-running-experiments)
5. [Experiment outputs](#5-experiment-outputs)
6. [License](#6-license)

---

## 1. Repository layout

```
STR/
├── config.py              Shared constants (seeds, batch sizes, data paths)
├── pyproject.toml         pdm project + dependencies
├── data/
│   ├── raw/{yahoo,smd}/   Raw benchmark data
│   └── consolidated/      Parquet versions of Yahoo (produced by the notebook)
├── notebooks/             Yahoo-data-consolidation.ipynb
├── scenarios/             Yahoo runner registry (DATASETS, MODELS, kwargs)
├── src/
│   ├── models/autoencoder/   GRU / LSTM / TCN × standard / variational
│   ├── regimes/              10 CL strategies (Naive, Cumulative, STE,
│   │                         EWC, LwF, RB, RES, GDumb, DER++, STR)
│   └── utils/                Data loading, reproducibility
├── scripts/
│   ├── run_experiments_yahoo.py   Yahoo!~A1/A3/A4 runner
│   └── run_experiments_smd.py     SMD runner
└── experiments/           Result CSVs + per-run artifacts (see §5)
```

## 2. Environment setup

Python 3.11 + [pdm](https://pdm-project.org/) are required.

```bash
# install dependencies into a local .venv
pdm install

# (optional) activate the shell
$(pdm venv activate)
```

All commands below assume `pdm run` prefixes, which resolves to the project virtualenv automatically.

## 3. Data

| Benchmark | Source | Used directly from |
|-----------|--------|--------------------|
| Yahoo!~A1, A2, A3, A4 | `data/raw/yahoo/` (CSV per series) | `data/consolidated/yahoo_a{1..4}.parquet` |
| SMD | `data/raw/smd/` (machine-{1,2,3}-{1..N}.txt) | `data/raw/smd/` (no consolidation step) |

### Yahoo - consolidation step (one-time)

Yahoo benchmarks ship as one CSV per series. The runner reads pre-consolidated parquet:

```bash
pdm run jupyter notebook notebooks/Yahoo-data-consolidation.ipynb
```

Execute all cells once to (re)generate `data/consolidated/yahoo_a{1..4}.parquet`. The parquet files are already in this repo, so this step is only needed if you re-extract the raw data or change the consolidation logic.

### SMD - used as-is

SMD is multivariate (38 features) and read directly from `data/raw/smd/`. No consolidation needed.

## 4. Running experiments

There are two runners - one per benchmark family. Both are crash-safe (CSV is written after every run), resumable (`--resume`), and lightweight (~10 KB per run, no model weights persisted).

| Runner | Benchmarks | Uses `scenarios/`? |
|--------|------------|--------------------|
| `scripts/run_experiments_yahoo.py` | Yahoo!~A1, A3, A4 | yes (registry-driven) |
| `scripts/run_experiments_smd.py` | SMD | no (self-contained configs) |

### Yahoo runner

```bash
# Yahoo A1 - full grid (10 strategies × 9 architectures × OFAT configs)
pdm run python scripts/run_experiments_yahoo.py --dataset yahoo-a1

# inspect the planned grid without running anything
pdm run python scripts/run_experiments_yahoo.py --dataset yahoo-a1 --dry-run

# resume after a crash / partial run
pdm run python scripts/run_experiments_yahoo.py --dataset yahoo-a1 --resume

# repeat for Yahoo A3 / A4
pdm run python scripts/run_experiments_yahoo.py --dataset yahoo-a3
pdm run python scripts/run_experiments_yahoo.py --dataset yahoo-a4
```

### SMD runner

SMD partitions the 28 machines into 5 sequential concepts (cluster-based, domain-incremental):

```
C1 - cluster1 top-4   (1-6, 1-7, 1-1, 1-3)
C2 - cluster1 other-4 (1-2, 1-4, 1-5, 1-8)
C3 - cluster2 top-4   (2-2, 2-4, 2-9, 2-1)
C4 - cluster2 other-4 (2-5, 2-7, 2-6, 2-3)
C5 - cluster3 top-4   (3-8, 3-2, 3-10, 3-6)
```

```bash
# SMD - full grid
pdm run python scripts/run_experiments_smd.py

# dry-run / resume / single architecture / sanity-limit first N
pdm run python scripts/run_experiments_smd.py --dry-run
pdm run python scripts/run_experiments_smd.py --resume
pdm run python scripts/run_experiments_smd.py --model gru
pdm run python scripts/run_experiments_smd.py --limit 5
```

## 5. Experiment outputs

All runs land under `experiments/{benchmark}/`. The directory already contains the results used in the paper - re-running overwrites or appends via `--resume`.

| Path | Size | Role |
|------|------|------|
| `experiments/yahoo-a1/` | 74 MB | Primary Yahoo benchmark; main-text tables |
| `experiments/yahoo-a3/` | 31 MB | Appendix tables |
| `experiments/yahoo-a4/` | 31 MB | Appendix tables |
| `experiments/smd/` | 96 MB | Primary SMD benchmark; main-text tables |
| `experiments/yahoo-a1_buffer_profile/` | 3.8 MB | §5.7.3 buffer-profile tables + PCA figure |
| `experiments/smd_buffer_profile/` | 181 MB | §5.7.3 buffer-profile tables + PCA figure |

### Per-benchmark structure

```
experiments/{benchmark}/
├── summary.csv                One row per run - all metrics, configs, CO₂
├── multiseed_summary.csv      (yahoo-a1, smd) extra seeds for headline cells
├── fwt.csv                    Forward-transfer values (recomputed)
└── {arch}/{run_id}/
    ├── params.json            Resolved hyperparameters for this run
    ├── codecarbon.csv         CodeCarbon emissions log
    └── *.parquet              Per-concept metric trajectories (for heatmaps)
```

`summary.csv` is the single source of truth for all paper tables and most figures. The per-run parquets feed the per-concept heatmap galleries.

## 6. License

MIT - see [`LICENSE`](LICENSE).
