import argparse
import contextlib
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from codecarbon import EmissionsTracker
from torch import optim
from tqdm import tqdm

from scenarios.registry import DATASETS, MODELS, TRAINING_KWARGS, CONTINUAL_KWARGS
from src.regimes.cumulative import CumulativeTrainingRegime
from src.regimes.der_pp import DERPlusPlusRegime
from src.regimes.ewc import EWCRegime
from src.regimes.experts import SingleTaskExpertsRegime
from src.regimes.gdumb import GDumbRegime
from src.regimes.lwf import LwFRegime
from src.regimes.naive import NaiveTrainingRegime
from src.regimes.random_buffer import RandomBufferRegime
from src.regimes.reservoir import ReservoirBufferRegime
from src.regimes.strb import STRBRegime
from src.utils.data import get_temporal_concept_datasets
from src.utils.reproducibility import ensure_reproducibility

SEED = TRAINING_KWARGS["seed"]

# Strategy defaults
STRB_DEFAULTS = dict(buffer_size=50, novelty_threshold=0.20, retention_threshold=0.9, replay_ratio=0.5)
REPLAY_DEFAULTS = dict(buffer_size=50, replay_ratio=0.5)
EWC_DEFAULTS = dict(lambda_ewc=1000.0)
LWF_DEFAULTS = dict(alpha=1.0, temperature=2.0)
DERPP_DEFAULTS = dict(buffer_size=50, replay_ratio=0.5, alpha=0.5, temperature=0.5)


# OFAT generation
def generate_configs(seq_lens, strides_map, split_sizes):
    configs = []

    # Default data params
    default_sl = 12
    default_stride = 12
    default_split = 336

    # Baselines (no params)
    for regime in ["naive", "cumulative", "experts"]:
        configs.append(dict(
            regime=regime, seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **_empty_params(),
        ))

    for overrides in _ofat(STRB_DEFAULTS, {
        "buffer_size": [50, 100, 250, 500, 1000, 2000, 4000],
        "novelty_threshold": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
        "retention_threshold": [0.3, 0.5, 0.7, 0.9, 0.95],
        "replay_ratio": [None, 0.3, 0.5, 0.7, 0.9],
    }):
        configs.append(dict(
            regime="strb", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # Random Buffer
    for overrides in _ofat(REPLAY_DEFAULTS, {
        "buffer_size": [50, 100, 250, 500, 1000],
        "replay_ratio": [None, 0.3, 0.5],
    }):
        configs.append(dict(
            regime="random-buffer", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # Reservoir Buffer
    for overrides in _ofat(REPLAY_DEFAULTS, {
        "buffer_size": [50, 100, 250, 500, 1000],
        "replay_ratio": [None, 0.3, 0.5],
    }):
        configs.append(dict(
            regime="reservoir", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # EWC
    for overrides in _ofat(EWC_DEFAULTS, {
        "lambda_ewc": [100, 500, 1000, 5000, 10000],
    }):
        configs.append(dict(
            regime="ewc", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # LwF
    for overrides in _ofat(LWF_DEFAULTS, {
        "alpha": [0.1, 0.5, 1.0, 2.0],
        "temperature": [1.0, 2.0, 5.0],
    }):
        configs.append(dict(
            regime="lwf", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # GDumb
    for overrides in _ofat(REPLAY_DEFAULTS, {
        "buffer_size": [50, 100, 250, 500, 1000],
        "replay_ratio": [None, 0.3, 0.5],
    }):
        configs.append(dict(
            regime="gdumb", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # DER++
    for overrides in _ofat(DERPP_DEFAULTS, {
        "buffer_size": [50, 100, 250, 500, 1000],
        "replay_ratio": [None, 0.3, 0.5],
        "alpha": [0.1, 0.5, 1.0],
        "temperature": [0.1, 0.5, 1.0],
    }):
        configs.append(dict(
            regime="derpp", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B1", **overrides,
        ))

    # Block 2: Seq length ablation (default params, all strategies)
    for sl in seq_lens:
        if sl == default_sl:
            continue  # already in B1
        for regime in ["naive", "cumulative", "experts", "strb", "random-buffer",
                       "reservoir", "ewc", "lwf", "gdumb", "derpp"]:
            configs.append(dict(
                regime=regime, seq_len=sl, stride=sl,
                split_size=default_split, block="B2",
                **_defaults_for(regime),
            ))

    # Block 3: Stride ablation
    for sl, strides in strides_map.items():
        for stride in strides:
            if stride == sl:
                continue  # non-overlapping already in B1/B2
            for regime in ["naive", "cumulative", "experts", "strb", "random-buffer",
                           "reservoir", "ewc", "lwf"]:
                configs.append(dict(
                    regime=regime, seq_len=sl, stride=stride,
                    split_size=default_split, block="B3",
                    **_defaults_for(regime),
                ))

    # Block 4: STRB buffer x novelty x seq_len (non-overlapping only)
    bs_values = [50, 250, 1000]
    nt_values = [0.05, 0.20, 0.35]
    all_sl_stride = [(sl, sl) for sl in seq_lens]  # non-overlapping only for STRB grid

    for bs, nt in itertools.product(bs_values, nt_values):
        for sl, stride in all_sl_stride:
            configs.append(dict(
                regime="strb", seq_len=sl, stride=stride,
                split_size=default_split, block="B4",
                buffer_size=bs, novelty_threshold=nt,
                retention_threshold=STRB_DEFAULTS["retention_threshold"],
                replay_ratio=STRB_DEFAULTS["replay_ratio"],
                lambda_ewc=None, alpha=None, temperature=None,
            ))

    # Block 5: Concept granularity
    for split in split_sizes:
        if split == default_split:
            continue
        for regime in ["naive", "cumulative", "experts", "strb", "random-buffer",
                       "reservoir", "ewc", "lwf", "gdumb", "derpp"]:
            configs.append(dict(
                regime=regime, seq_len=default_sl, stride=default_stride,
                split_size=split, block="B5",
                **_defaults_for(regime),
            ))

    # Block 6: STRB retention x novelty cross-grid
    re_values = [0.3, 0.5, 0.7, 0.9, 0.95]
    nv_values = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    for re_t, nv_t in itertools.product(re_values, nv_values):
        configs.append(dict(
            regime="strb", seq_len=default_sl, stride=default_stride,
            split_size=default_split, block="B6",
            buffer_size=STRB_DEFAULTS["buffer_size"],
            novelty_threshold=nv_t,
            retention_threshold=re_t,
            replay_ratio=STRB_DEFAULTS["replay_ratio"],
            lambda_ewc=None, alpha=None, temperature=None,
        ))

    # Deduplicate by run_id
    seen, deduped = set(), []
    for c in configs:
        rid = config_to_id(c)
        if rid not in seen:
            seen.add(rid)
            deduped.append(c)

    return deduped


def _empty_params():
    return dict(buffer_size=None, novelty_threshold=None, retention_threshold=None,
                replay_ratio=None, lambda_ewc=None, alpha=None, temperature=None)


def _defaults_for(regime):
    if regime == "strb":
        return dict(**STRB_DEFAULTS, lambda_ewc=None, alpha=None, temperature=None)
    elif regime in ("random-buffer", "reservoir", "gdumb"):
        return dict(**REPLAY_DEFAULTS, novelty_threshold=None, retention_threshold=None,
                    lambda_ewc=None, alpha=None, temperature=None)
    elif regime == "ewc":
        return dict(**EWC_DEFAULTS, buffer_size=None, novelty_threshold=None,
                    retention_threshold=None, replay_ratio=None, alpha=None, temperature=None)
    elif regime == "lwf":
        return dict(**LWF_DEFAULTS, buffer_size=None, novelty_threshold=None,
                    retention_threshold=None, replay_ratio=None, lambda_ewc=None)
    elif regime == "derpp":
        return dict(**DERPP_DEFAULTS, novelty_threshold=None, retention_threshold=None,
                    lambda_ewc=None)
    else:  # naive, cumulative, experts
        return _empty_params()


def _ofat(defaults, grid):
    """Generate OFAT configs: default + vary one param at a time."""
    configs = [dict(defaults, **_fill_none(defaults))]
    for param, values in grid.items():
        for v in values:
            if v != defaults.get(param):
                c = dict(defaults)
                c[param] = v
                c.update(_fill_none(c))
                configs.append(c)
    seen, unique = set(), []
    for c in configs:
        key = tuple(sorted(c.items()))
        if key not in seen:
            seen.add(key)
            unique.append(c)
    return unique


def _fill_none(d):
    all_keys = ["buffer_size", "novelty_threshold", "retention_threshold",
                "replay_ratio", "lambda_ewc", "alpha", "temperature"]
    return {k: d.get(k) for k in all_keys}


_INT_HPARAMS = {"buffer_size", "lambda_ewc"}


def config_to_id(c):
    sl = int(c["seq_len"]); st = int(c["stride"]); sp = int(c["split_size"])
    parts = [f"r-{c['regime']}", f"sl{sl}", f"st{st}", f"sp{sp}"]
    for k in ["buffer_size", "novelty_threshold", "retention_threshold",
              "replay_ratio", "lambda_ewc", "alpha", "temperature"]:
        v = c.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k in _INT_HPARAMS and isinstance(v, float) and float(v).is_integer():
            v = int(v)
        parts.append(f"{k[:2]}{v}")
    return "_".join(parts)


def compute_metrics(artifacts_path, n_concepts):
    df = pd.read_parquet(artifacts_path / "df_classification_stats.parquet")
    value_col = "roc_auc" if "roc_auc" in df.columns else "roc_auc_macro"
    pivot = df.pivot(index="current_train_concept", columns="reference_test_concept",
                     values=value_col)
    n = len(pivot)

    bwt_vals = [pivot.iloc[t, k] - pivot.iloc[k, k] for t in range(1, n) for k in range(t)]
    bwt = float(np.nanmean(bwt_vals)) if bwt_vals else 0.0

    fwt_vals = [pivot.iloc[i - 1, i] - 0.5 for i in range(1, n)]
    fwt = float(np.nanmean(fwt_vals)) if fwt_vals else 0.0

    diag = [pivot.iloc[i, i] for i in range(n)]
    final = pivot.iloc[-1, :].tolist()
    n_valid_diag = int(np.sum(~np.isnan(diag)))
    n_valid_bwt_pairs = int(np.sum(~np.isnan(bwt_vals))) if bwt_vals else 0
    n_valid_af_pairs = 0  # filled below

    cc_path = artifacts_path / "codecarbon.csv"
    co2 = 0.0
    energy_kwh = 0.0
    cpu_energy_kwh = 0.0
    ram_energy_kwh = 0.0
    cc_duration = 0.0
    if cc_path.exists():
        try:
            cc = pd.read_csv(cc_path)
            co2 = cc["emissions"].sum() if "emissions" in cc.columns else 0.0
            energy_kwh = cc["energy_consumed"].sum() if "energy_consumed" in cc.columns else 0.0
            cpu_energy_kwh = cc["cpu_energy"].sum() if "cpu_energy" in cc.columns else 0.0
            ram_energy_kwh = cc["ram_energy"].sum() if "ram_energy" in cc.columns else 0.0
            cc_duration = cc["duration"].sum() if "duration" in cc.columns else 0.0
        except Exception:
            pass

    # average forgetting
    af_vals = []
    for j in range(n):
        max_so_far = pivot.iloc[j, j]
        for t in range(j + 1, n):
            af_vals.append(max_so_far - pivot.iloc[t, j])
    af = float(np.nanmean(af_vals)) if af_vals else 0.0
    n_valid_af_pairs = int(np.sum(~np.isnan(af_vals))) if af_vals else 0

    total_epochs = 0
    ts_path = artifacts_path / "df_training_stats.parquet"
    if ts_path.exists():
        ts = pd.read_parquet(ts_path)
        total_epochs = int(ts.groupby("current_train_concept")["epoch"].max().sum() + n)

    return dict(
        bwt=bwt, fwt=fwt, af=af,
        avg_diagonal_auc=float(np.nanmean(diag)) if n_valid_diag else float('nan'),
        final_avg_auc=float(np.nanmean(final)) if np.any(~np.isnan(final)) else float('nan'),
        peak_auc=float(np.nanmax(pivot.values)) if np.any(~np.isnan(pivot.values)) else float('nan'),
        n_valid_diag=n_valid_diag,
        n_valid_bwt_pairs=n_valid_bwt_pairs,
        n_valid_af_pairs=n_valid_af_pairs,
        **{f"diag_c{i}": (round(v, 4) if not np.isnan(v) else float('nan')) for i, v in enumerate(diag)},
        **{f"final_c{i}": (round(v, 4) if not np.isnan(v) else float('nan')) for i, v in enumerate(final)},
        co2_kg=co2,
        energy_kwh=energy_kwh,
        cpu_energy_kwh=cpu_energy_kwh,
        ram_energy_kwh=ram_energy_kwh,
        cc_duration_s=round(cc_duration, 1),
        total_epochs=total_epochs,
    )


def run_single(config, ds_key, mdl_key, artifacts_path):
    ensure_reproducibility(SEED)
    ds = DATASETS[ds_key]
    mdl = MODELS[mdl_key]

    sl = config["seq_len"]
    ae_config = mdl.ae_config.model_copy(update={"seq_len": sl})

    df = ds.read_fn()
    train_ds, test_ds = get_temporal_concept_datasets(
        df=df, value_column=ds.value_column, label_column=ds.label_column,
        split_size=config["split_size"], sequence_length=sl,
        train_size=0.7, train_sequence_step=config["stride"], test_sequence_step=1,
    )

    model = mdl.model_cls(ae_config)
    base_kwargs = dict(
        autoencoder=model, variational=mdl.variational, optimizer_cls=optim.Adam,
        criterion_cls=mdl.criterion_cls, criterion_fn=mdl.criterion_fn,
        train_datasets=train_ds, test_datasets=test_ds,
        artifacts_path=artifacts_path, **TRAINING_KWARGS,
    )

    regime = config["regime"]
    if regime == "naive":
        r = NaiveTrainingRegime(**base_kwargs)
    elif regime == "cumulative":
        r = CumulativeTrainingRegime(**base_kwargs)
    elif regime == "experts":
        r = SingleTaskExpertsRegime(**base_kwargs)
    elif regime == "strb":
        r = STRBRegime(**base_kwargs,
                       buffer_size=config["buffer_size"],
                       novelty_threshold=config["novelty_threshold"],
                       retention_threshold=config["retention_threshold"],
                       replay_ratio=config["replay_ratio"])
    elif regime == "random-buffer":
        r = RandomBufferRegime(**base_kwargs,
                               buffer_size=config["buffer_size"],
                               replay_ratio=config["replay_ratio"])
    elif regime == "reservoir":
        r = ReservoirBufferRegime(**base_kwargs,
                                  buffer_size=config["buffer_size"],
                                  replay_ratio=config["replay_ratio"])
    elif regime == "ewc":
        r = EWCRegime(**base_kwargs, lambda_ewc=config["lambda_ewc"])
    elif regime == "lwf":
        r = LwFRegime(**base_kwargs, alpha=config["alpha"],
                      temperature=config["temperature"])
    elif regime == "gdumb":
        r = GDumbRegime(**base_kwargs,
                        buffer_size=config["buffer_size"],
                        replay_ratio=config["replay_ratio"])
    elif regime == "derpp":
        # DER++ uses 'temperature' column to store beta (distillation weight)
        r = DERPlusPlusRegime(**base_kwargs,
                              buffer_size=config["buffer_size"],
                              replay_ratio=config["replay_ratio"],
                              alpha=config["alpha"],
                              beta=config["temperature"])
    else:
        raise ValueError(f"Unknown regime: {regime}")

    # CodeCarbon
    cc_path = artifacts_path / "codecarbon.csv"
    tracker = EmissionsTracker(
        project_name=f"{ds_key}-{mdl_key}-{regime}",
        output_file=str(cc_path),
        log_level="error",
        allow_multiple_runs=True,
    )

    import warnings
    t0 = time.time()
    try:
        tracker.start()
        has_tracker = True
    except Exception:
        has_tracker = False
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull), \
         contextlib.redirect_stderr(devnull), warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r.start()
    if has_tracker:
        tracker.stop()
    r.stop()
    elapsed = time.time() - t0

    # Save params
    with open(artifacts_path / "params.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    n_concepts = len(train_ds)
    metrics = compute_metrics(artifacts_path, n_concepts)

    return {
        "dataset": ds_key, "model": mdl_key,
        **config,
        **metrics,
        "n_concepts": n_concepts,
        "elapsed_s": round(elapsed, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=list(DATASETS))
    parser.add_argument("--model", default="all", choices=list(MODELS) + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]

    seq_lens = [6, 8, 10, 12, 16, 24]
    strides_map = {8: [8, 4], 12: [12, 6], 24: [24, 12]}  # non-overlapping + 50% overlap
    split_sizes = [168, 336]

    configs = generate_configs(seq_lens, strides_map, split_sizes)
    all_runs = [(mdl_key, c) for mdl_key in model_keys for c in configs]
    total = len(all_runs)

    out_dir = Path("experiments") / args.dataset
    out_csv = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Experiment runner: {args.dataset}")
    print(f"Models: {model_keys}")
    print(f"Configs per model: {len(configs)}")
    print(f"Total runs: {total}")
    print(f"Output: {out_csv}")
    print(f"Est. time: {total * 40 / 3600:.1f} hours\n")

    if args.dry_run:
        from collections import Counter
        regime_counts = Counter(c["regime"] for _, c in all_runs)
        block_counts = Counter(c["block"] for _, c in all_runs)
        print("By regime:")
        for r, n in sorted(regime_counts.items()):
            print(f"  {r}: {n}")
        print(f"\nBy block:")
        for b, n in sorted(block_counts.items()):
            print(f"  {b}: {n}")
        return

    # Load existing results for resume
    results = []
    existing_ids = set()
    if args.resume and out_csv.exists():
        existing = pd.read_csv(out_csv)
        results = existing.to_dict("records")
        existing_ids = set(
            f"{r['model']}_{config_to_id(r)}" for r in results
        )
        print(f"Resuming: {len(results)} existing results\n")

    # Filter out already-completed runs
    pending = [(mdl_key, config) for mdl_key, config in all_runs
               if f"{mdl_key}_{config_to_id(config)}" not in existing_ids]

    if len(pending) < total:
        print(f"Skipping {total - len(pending)} completed runs\n")

    pbar = tqdm(pending, desc="Experiments", unit="run",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for mdl_key, config in pbar:
        run_id = config_to_id(config)
        artifacts = out_dir / mdl_key / run_id
        os.makedirs(artifacts, exist_ok=True)

        pbar.set_postfix_str(f"{mdl_key}/{config['regime']}", refresh=True)

        try:
            row = run_single(config, args.dataset, mdl_key, artifacts)
            row["run_id"] = run_id
            results.append(row)

            pbar.set_postfix_str(
                f"{mdl_key}/{config['regime']} BWT={row['bwt']:+.3f} AUC={row['final_avg_auc']:.3f}",
                refresh=True)

            # Save incrementally
            pd.DataFrame(results).to_csv(out_csv, index=False)

        except Exception as e:
            tqdm.write(f"ERROR [{mdl_key}/{run_id}]: {e}")

    pbar.close()
    print(f"\nDone: {len(results)} runs. Results: {out_csv}")


if __name__ == "__main__":
    main()
