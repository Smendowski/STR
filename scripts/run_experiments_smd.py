import argparse
import contextlib
import itertools
import json
import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from sklearn.preprocessing import MinMaxScaler, PowerTransformer
from torch import optim
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    EARLY_STOPPING_PATIENCE, LEARNING_RATE, SEED, SEQUENCE_LENGTH,
    SMD_BATCH_SIZE, SMD_EARLY_STOPPING_MIN_DELTA, SMD_EPOCHS,
)
from src.models.autoencoder.config import (
    ActivationLayerConfig, AutoencoderConfig, Conv1dLayerConfig,
    ConvTranspose1dLayerConfig, DecoderConfig, EncoderConfig,
    GRULayerConfig, LSTMLayerConfig,
)
from src.models.autoencoder.standard.gru import GRUAutoencoder
from src.models.autoencoder.standard.lstm import LSTMAutoencoder
from src.models.autoencoder.standard.tcn import TCNAutoencoder
from src.models.autoencoder.variational.gru import GRUVariationalAutoencoder
from src.models.autoencoder.variational.lstm import LSTMVariationalAutoencoder
from src.models.autoencoder.variational.tcn import TCNVariationalAutoencoder
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
from src.utils.data.concept import TemporalConcept, TemporalConceptDataset
from src.utils.reproducibility import ensure_reproducibility


SMD_ROOT = ROOT / "data" / "raw" / "smd"
F_IN = 38
HIDDEN = 16
LATENT = 8
TRAIN_LEN = 23_000
TEST_LEN = 23_688

DOMAIN_DEFINITIONS = [
    ("C1_cluster1_top",   ["machine-1-6", "machine-1-7", "machine-1-1", "machine-1-3"]),
    ("C2_cluster1_other", ["machine-1-2", "machine-1-4", "machine-1-5", "machine-1-8"]),
    ("C3_cluster2_top",   ["machine-2-2", "machine-2-4", "machine-2-9", "machine-2-1"]),
    ("C4_cluster2_other", ["machine-2-5", "machine-2-7", "machine-2-6", "machine-2-3"]),
    ("C5_cluster3_top",   ["machine-3-8", "machine-3-2", "machine-3-10", "machine-3-6"]),
]

ARCHITECTURES = ["gru", "lstm", "tcn", "vgru", "vlstm", "vtcn",
                 "vgru-kl", "vlstm-kl", "vtcn-kl"]

KL_WARMUP_EPOCHS = 20
KL_WARMUP_TARGET_BETA = 1.0

STRB_DEFAULTS = dict(buffer_size=500, novelty_threshold=0.20,
                     retention_threshold=0.9, replay_ratio=0.5)
REPLAY_DEFAULTS = dict(buffer_size=500, replay_ratio=0.5)
EWC_DEFAULTS = dict(lambda_ewc=1000.0)
LWF_DEFAULTS = dict(alpha=1.0, temperature=2.0)
DERPP_DEFAULTS = dict(buffer_size=500, replay_ratio=0.5, alpha=0.5, temperature=0.5)  # temperature reused as beta

DEFAULT_SL = 24
DEFAULT_STRIDE = 12

SEQ_LENS = [6, 8, 10, 12, 16, 24, 48]

STRIDES_MAP = {
    8: [4],
    12: [6],
    24: [12],
    48: [24],
}

def make_arch_config(arch: str, seq_len: int) -> AutoencoderConfig:
    """Build an AutoencoderConfig per architecture for multivariate input."""
    common_seq_len = seq_len

    def gru_layers(in_size, hidden, latent):
        return EncoderConfig(layers=[
            GRULayerConfig(kwargs={"input_size": in_size, "hidden_size": hidden,
                                    "num_layers": 1, "batch_first": True}),
            ActivationLayerConfig(cls=nn.Tanh),
            GRULayerConfig(kwargs={"input_size": hidden, "hidden_size": latent,
                                    "num_layers": 1, "batch_first": True}),
        ])

    def gru_dec(latent, hidden, out_size):
        return DecoderConfig(layers=[
            GRULayerConfig(kwargs={"input_size": latent, "hidden_size": hidden,
                                    "num_layers": 1, "batch_first": True}),
            ActivationLayerConfig(cls=nn.Tanh),
            GRULayerConfig(kwargs={"input_size": hidden, "hidden_size": out_size,
                                    "num_layers": 1, "batch_first": True}),
            ActivationLayerConfig(cls=nn.Tanh),
        ])

    def lstm_layers(in_size, hidden, latent):
        return EncoderConfig(layers=[
            LSTMLayerConfig(kwargs={"input_size": in_size, "hidden_size": hidden,
                                     "num_layers": 1, "batch_first": True}),
            ActivationLayerConfig(cls=nn.Tanh),
            LSTMLayerConfig(kwargs={"input_size": hidden, "hidden_size": latent,
                                     "num_layers": 1, "batch_first": True}),
        ])

    def lstm_dec(latent, hidden, out_size):
        return DecoderConfig(layers=[
            LSTMLayerConfig(kwargs={"input_size": latent, "hidden_size": hidden,
                                     "num_layers": 1, "batch_first": True}),
            ActivationLayerConfig(cls=nn.Tanh),
            LSTMLayerConfig(kwargs={"input_size": hidden, "hidden_size": out_size,
                                     "num_layers": 1, "batch_first": True}),
            ActivationLayerConfig(cls=nn.Tanh),
        ])

    def tcn_layers(in_size, hidden, latent):
        return EncoderConfig(layers=[
            Conv1dLayerConfig(kwargs={"in_channels": in_size, "out_channels": hidden,
                                        "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1}),
            ActivationLayerConfig(cls=nn.Tanh),
            Conv1dLayerConfig(kwargs={"in_channels": hidden, "out_channels": latent,
                                        "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1}),
            ActivationLayerConfig(cls=nn.Tanh),
        ])

    def tcn_dec(latent, hidden, out_size):
        return DecoderConfig(layers=[
            ConvTranspose1dLayerConfig(kwargs={"in_channels": latent, "out_channels": hidden,
                                                "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1}),
            ActivationLayerConfig(cls=nn.Tanh),
            ConvTranspose1dLayerConfig(kwargs={"in_channels": hidden, "out_channels": out_size,
                                                "kernel_size": 3, "stride": 1, "padding": 1, "dilation": 1}),
            ActivationLayerConfig(cls=nn.Tanh),
        ])

    if arch in ("gru", "vgru", "vgru-kl"):
        return AutoencoderConfig(seq_len=common_seq_len,
                                  encoder=gru_layers(F_IN, HIDDEN, LATENT),
                                  decoder=gru_dec(LATENT, HIDDEN, F_IN))
    if arch in ("lstm", "vlstm", "vlstm-kl"):
        return AutoencoderConfig(seq_len=common_seq_len,
                                  encoder=lstm_layers(F_IN, HIDDEN, LATENT),
                                  decoder=lstm_dec(LATENT, HIDDEN, F_IN))
    if arch in ("tcn", "vtcn", "vtcn-kl"):
        return AutoencoderConfig(seq_len=common_seq_len,
                                  encoder=tcn_layers(F_IN, HIDDEN, LATENT),
                                  decoder=tcn_dec(LATENT, HIDDEN, F_IN))
    raise ValueError(f"Unknown arch: {arch}")


def instantiate_model(arch: str, config: AutoencoderConfig):
    return {
        "gru": GRUAutoencoder, "lstm": LSTMAutoencoder, "tcn": TCNAutoencoder,
        "vgru": GRUVariationalAutoencoder, "vlstm": LSTMVariationalAutoencoder,
        "vtcn": TCNVariationalAutoencoder,
        "vgru-kl": GRUVariationalAutoencoder,
        "vlstm-kl": LSTMVariationalAutoencoder,
        "vtcn-kl": TCNVariationalAutoencoder,
    }[arch](config)


def is_variational(arch: str) -> bool:
    return arch.startswith("v")


def _make_warmup_loss_cls():
    from src.models.autoencoder.loss import WarmupBetaVariationalMSELoss
    target_beta = KL_WARMUP_TARGET_BETA
    warmup_epochs = KL_WARMUP_EPOCHS

    class _WarmupBeta(WarmupBetaVariationalMSELoss):
        def __init__(self, reduction: str = "mean"):
            super().__init__(reduction=reduction,
                             target_beta=target_beta,
                             warmup_epochs=warmup_epochs)

    _WarmupBeta.__name__ = f"WarmupBetaVAELoss_b{target_beta}_w{warmup_epochs}"
    return _WarmupBeta


def criterion_for(arch: str):
    from src.models.autoencoder.loss import VariationalMSELoss
    if arch.endswith("-kl"):
        return _make_warmup_loss_cls(), nn.functional.mse_loss
    if is_variational(arch):
        return VariationalMSELoss, nn.functional.mse_loss
    return nn.MSELoss, nn.functional.mse_loss


_machine_cache: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}


def load_machine_cached(name: str):
    if name in _machine_cache:
        return _machine_cache[name]
    train = np.loadtxt(SMD_ROOT / "train" / f"{name}.txt", delimiter=",", dtype=np.float32)
    test = np.loadtxt(SMD_ROOT / "test" / f"{name}.txt", delimiter=",", dtype=np.float32)
    labels = np.loadtxt(SMD_ROOT / "test_label" / f"{name}.txt", dtype=np.int64)
    _machine_cache[name] = (train, test, labels)
    return train, test, labels


def _make_4d_mv(arr_3d: np.ndarray, seq_len: int, step: int) -> np.ndarray:
    n_chan, T, F = arr_3d.shape
    n_seq = (T - seq_len) // step + 1
    out = np.empty((n_chan, n_seq, seq_len, F), dtype=np.float32)
    for c in range(n_chan):
        for s in range(n_seq):
            out[c, s] = arr_3d[c, s * step:s * step + seq_len]
    return out


def _make_4d_labels(arr_2d: np.ndarray, seq_len: int, step: int, f_dim: int) -> np.ndarray:
    n_chan, T = arr_2d.shape
    n_seq = (T - seq_len) // step + 1
    out = np.empty((n_chan, n_seq, seq_len, f_dim), dtype=np.int64)
    for c in range(n_chan):
        for s in range(n_seq):
            seg = arr_2d[c, s * step:s * step + seq_len]
            out[c, s] = np.broadcast_to(seg[:, None], (seq_len, f_dim))
    return out


def build_concepts(seq_len: int, train_step: int):
    train_dss, test_dss = [], []
    for label, machines in DOMAIN_DEFINITIONS:
        trains, tests, lbls = [], [], []
        for m in machines:
            tr, te, lb = load_machine_cached(m)
            trains.append(tr[:TRAIN_LEN])
            tests.append(te[:TEST_LEN])
            lbls.append(lb[:TEST_LEN])
        train_block = np.stack(trains, axis=0)
        test_block = np.stack(tests, axis=0)
        label_block = np.stack(lbls, axis=0)

        F = train_block.shape[2]
        train_flat = train_block.reshape(-1, F)
        scaler = MinMaxScaler().fit(train_flat)
        pt = PowerTransformer(method="yeo-johnson").fit(scaler.transform(train_flat))

        def _scale(x):
            return np.clip(pt.transform(scaler.transform(x.reshape(-1, F))) / 3,
                            -1, 1).astype(np.float32).reshape(x.shape)

        train_scaled = _scale(train_block)
        test_scaled = _scale(test_block)

        train_data_4d = _make_4d_mv(train_scaled, seq_len, train_step)
        zeros = np.zeros((train_block.shape[0], train_block.shape[1]), dtype=np.int64)
        train_labels_4d = _make_4d_labels(zeros, seq_len, train_step, F)
        test_data_4d = _make_4d_mv(test_scaled, seq_len, seq_len)  # non-overlapping
        test_labels_4d = _make_4d_labels(label_block, seq_len, seq_len, F)

        train_t = torch.from_numpy(train_data_4d)
        train_l = torch.from_numpy(train_labels_4d)
        test_t = torch.from_numpy(test_data_4d)
        test_l = torch.from_numpy(test_labels_4d)

        train_ds = TemporalConceptDataset(TemporalConcept(
            name=f"train_{label}", data=train_t, labels=train_l,
            timeline=np.arange(TEST_LEN)))
        test_ds = TemporalConceptDataset(TemporalConcept(
            name=f"test_{label}", data=test_t, labels=test_l,
            timeline=np.arange(TEST_LEN)))
        train_ds.step = train_step
        test_ds.step = seq_len
        train_dss.append(train_ds)
        test_dss.append(test_ds)
    return train_dss, test_dss


def _empty_params():
    return dict(buffer_size=None, novelty_threshold=None, retention_threshold=None,
                replay_ratio=None, lambda_ewc=None, alpha=None, temperature=None)


def _defaults_for(regime):
    if regime == "strb":
        return dict(**STRB_DEFAULTS, lambda_ewc=None, alpha=None, temperature=None)
    if regime in ("random-buffer", "reservoir", "gdumb"):
        return dict(**REPLAY_DEFAULTS, novelty_threshold=None, retention_threshold=None,
                    lambda_ewc=None, alpha=None, temperature=None)
    if regime == "ewc":
        return dict(**EWC_DEFAULTS, buffer_size=None, novelty_threshold=None,
                    retention_threshold=None, replay_ratio=None, alpha=None, temperature=None)
    if regime == "lwf":
        return dict(**LWF_DEFAULTS, buffer_size=None, novelty_threshold=None,
                    retention_threshold=None, replay_ratio=None, lambda_ewc=None)
    if regime == "derpp":
        return dict(**DERPP_DEFAULTS, novelty_threshold=None, retention_threshold=None,
                    lambda_ewc=None)
    return _empty_params()


def _ofat(defaults, grid):
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
    keys = ["buffer_size", "novelty_threshold", "retention_threshold",
            "replay_ratio", "lambda_ewc", "alpha", "temperature"]
    return {k: d.get(k) for k in keys}


_INT_HPARAMS = {"buffer_size", "lambda_ewc"}


def config_to_id(c):
    sl = int(c["seq_len"]); st = int(c["stride"])
    parts = [f"r-{c['regime']}", f"sl{sl}", f"st{st}"]
    for k in ["buffer_size", "novelty_threshold", "retention_threshold",
              "replay_ratio", "lambda_ewc", "alpha", "temperature"]:
        v = c.get(k)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        if k in _INT_HPARAMS and isinstance(v, float) and float(v).is_integer():
            v = int(v)
        parts.append(f"{k[:2]}{v}")
    return "_".join(parts)


def generate_configs():
    configs = []

    # Block 1: Strategy OFAT at default seq/stride
    for regime in ["naive", "cumulative", "experts"]:
        configs.append(dict(regime=regime, seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                            block="B1", **_empty_params()))

    for overrides in _ofat(STRB_DEFAULTS, {
        "buffer_size": [100, 250, 500, 1000, 2000, 4000],
        "novelty_threshold": [0.01, 0.05, 0.10, 0.20, 0.35, 0.50],
        "retention_threshold": [0.3, 0.5, 0.7, 0.9, 0.95],
        "replay_ratio": [0.3, 0.5, 0.7, 0.9],
    }):
        configs.append(dict(regime="strb", seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                            block="B1", **overrides))

    for regime in ["random-buffer", "reservoir"]:
        for overrides in _ofat(REPLAY_DEFAULTS, {
            "buffer_size": [100, 250, 500, 1000, 2000, 4000],
            "replay_ratio": [0.3, 0.5, 0.7],
        }):
            configs.append(dict(regime=regime, seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                                block="B1", **overrides))

    for overrides in _ofat(EWC_DEFAULTS, {
        "lambda_ewc": [100, 500, 1000, 5000, 10000],
    }):
        configs.append(dict(regime="ewc", seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                            block="B1", **overrides))

    for overrides in _ofat(LWF_DEFAULTS, {
        "alpha": [0.5, 1.0, 2.0],
        "temperature": [1.0, 2.0, 4.0],
    }):
        configs.append(dict(regime="lwf", seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                            block="B1", **overrides))

    # GDumb
    for overrides in _ofat(REPLAY_DEFAULTS, {
        "buffer_size": [100, 250, 500, 1000, 2000, 4000],
        "replay_ratio": [0.3, 0.5, 0.7],
    }):
        configs.append(dict(regime="gdumb", seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                            block="B1", **overrides))

    # DER++
    for overrides in _ofat(DERPP_DEFAULTS, {
        "buffer_size": [100, 250, 500, 1000, 2000, 4000],
        "replay_ratio": [0.3, 0.5, 0.7],
        "alpha": [0.1, 0.5, 1.0],
        "temperature": [0.1, 0.5, 1.0],
    }):
        configs.append(dict(regime="derpp", seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
                            block="B1", **overrides))

    # Block 2: Seq length ablation at default hparams (non-overlapping windows)
    for sl in SEQ_LENS:
        if sl == DEFAULT_SL:
            continue
        for regime in ["naive", "cumulative", "experts", "strb", "random-buffer",
                       "reservoir", "ewc", "lwf", "gdumb", "derpp"]:
            configs.append(dict(regime=regime, seq_len=sl, stride=sl,
                                block="B2", **_defaults_for(regime)))

    # Block 3: Stride ablation (50% overlap variants)
    for sl, strides in STRIDES_MAP.items():
        for stride in strides:
            if (sl, stride) == (DEFAULT_SL, DEFAULT_STRIDE):
                continue
            for regime in ["naive", "cumulative", "experts", "strb", "random-buffer",
                           "reservoir", "ewc", "lwf"]:
                configs.append(dict(regime=regime, seq_len=sl, stride=stride,
                                    block="B3", **_defaults_for(regime)))

    # Block 4: STRB buffer x novelty cross at all seq_lens (non-overlapping only)
    bs_values = [100, 500, 1000]
    nt_values = [0.05, 0.20, 0.35]
    for bs, nt in itertools.product(bs_values, nt_values):
        for sl in SEQ_LENS:
            if sl == DEFAULT_SL and bs == STRB_DEFAULTS["buffer_size"] and nt == STRB_DEFAULTS["novelty_threshold"]:
                continue
            configs.append(dict(
                regime="strb", seq_len=sl, stride=sl,
                block="B4",
                buffer_size=bs, novelty_threshold=nt,
                retention_threshold=STRB_DEFAULTS["retention_threshold"],
                replay_ratio=STRB_DEFAULTS["replay_ratio"],
                lambda_ewc=None, alpha=None, temperature=None,
            ))

    # Block 6: STRB retention x novelty cross-grid (interaction ablation)
    re_values = [0.3, 0.5, 0.7, 0.9, 0.95]
    nv_values = [0.01, 0.05, 0.10, 0.20, 0.35, 0.50]
    for re_t, nv_t in itertools.product(re_values, nv_values):
        configs.append(dict(
            regime="strb", seq_len=DEFAULT_SL, stride=DEFAULT_STRIDE,
            block="B6",
            buffer_size=STRB_DEFAULTS["buffer_size"],
            novelty_threshold=nv_t,
            retention_threshold=re_t,
            replay_ratio=STRB_DEFAULTS["replay_ratio"],
            lambda_ewc=None, alpha=None, temperature=None,
        ))

    # Deduplicate
    seen, deduped = set(), []
    for c in configs:
        rid = config_to_id(c)
        if rid not in seen:
            seen.add(rid)
            deduped.append(c)
    return deduped


def compute_metrics(artifacts_path: Path, n_concepts: int):
    df = pd.read_parquet(artifacts_path / "df_classification_stats.parquet")
    value_col = "roc_auc" if "roc_auc" in df.columns else "roc_auc_macro"
    pivot = df.pivot(index="current_train_concept",
                     columns="reference_test_concept", values=value_col)
    n = len(pivot)

    bwt_vals = [pivot.iloc[t, k] - pivot.iloc[k, k]
                for t in range(1, n) for k in range(t)]
    bwt = float(np.nanmean(bwt_vals)) if bwt_vals else 0.0

    # Forward Transfer (Lopez-Paz & Ranzato, 2017): zero-shot benefit of
    # having trained through concept i-1, evaluated on concept i, averaged
    # over all adjacent-concept transitions. b_i = 0.5 is the random
    # baseline for ROC-AUC on a single test concept.
    fwt_vals = [pivot.iloc[i - 1, i] - 0.5 for i in range(1, n)]
    fwt = float(np.nanmean(fwt_vals)) if fwt_vals else 0.0

    diag = [pivot.iloc[i, i] for i in range(n)]
    final = pivot.iloc[-1, :].tolist()
    n_valid_diag = int(np.sum(~np.isnan(diag)))
    n_valid_bwt_pairs = int(np.sum(~np.isnan(bwt_vals))) if bwt_vals else 0

    cc_path = artifacts_path / "codecarbon.csv"
    co2 = energy_kwh = cpu_e = ram_e = cc_dur = 0.0
    if cc_path.exists():
        try:
            cc = pd.read_csv(cc_path)
            co2 = cc["emissions"].sum() if "emissions" in cc.columns else 0.0
            energy_kwh = cc["energy_consumed"].sum() if "energy_consumed" in cc.columns else 0.0
            cpu_e = cc["cpu_energy"].sum() if "cpu_energy" in cc.columns else 0.0
            ram_e = cc["ram_energy"].sum() if "ram_energy" in cc.columns else 0.0
            cc_dur = cc["duration"].sum() if "duration" in cc.columns else 0.0
        except Exception:
            pass

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
        co2_kg=co2, energy_kwh=energy_kwh,
        cpu_energy_kwh=cpu_e, ram_energy_kwh=ram_e,
        cc_duration_s=round(cc_dur, 1),
        total_epochs=total_epochs,
    )


PROGRESS_FILE = ROOT / "experiments" / "smd" / "current_run_progress.txt"


class _Tee:
    def __init__(self, *sinks):
        self._sinks = sinks

    def write(self, s):
        for sink in self._sinks:
            try:
                sink.write(s)
                sink.flush()
            except Exception:
                pass

    def flush(self):
        for sink in self._sinks:
            try:
                sink.flush()
            except Exception:
                pass

def run_single(config, arch, artifacts_path):
    ensure_reproducibility(SEED)
    sl = config["seq_len"]
    stride = config["stride"]

    train_ds_list, test_ds_list = build_concepts(seq_len=sl, train_step=stride)

    ae_config = make_arch_config(arch, sl)
    model = instantiate_model(arch, ae_config)
    crit_cls, crit_fn = criterion_for(arch)

    base_kwargs = dict(
        autoencoder=model, variational=is_variational(arch),
        optimizer_cls=optim.Adam,
        criterion_cls=crit_cls, criterion_fn=crit_fn,
        train_datasets=train_ds_list, test_datasets=test_ds_list,
        artifacts_path=artifacts_path,
        learning_rate=LEARNING_RATE, epochs=SMD_EPOCHS, batch_size=SMD_BATCH_SIZE,
        sequence_len=sl, seed=SEED,
        early_stopping_patience=EARLY_STOPPING_PATIENCE,
        early_stopping_min_delta=SMD_EARLY_STOPPING_MIN_DELTA,
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
        r = DERPlusPlusRegime(**base_kwargs,
                              buffer_size=config["buffer_size"],
                              replay_ratio=config["replay_ratio"],
                              alpha=config["alpha"],
                              beta=config["temperature"])
    else:
        raise ValueError(f"Unknown regime: {regime}")

    cc_path = artifacts_path / "codecarbon.csv"
    tracker = EmissionsTracker(
        project_name=f"smd-{arch}-{regime}",
        output_file=str(cc_path),
        log_level="error",
        allow_multiple_runs=True,
    )

    t0 = time.time()
    try:
        tracker.start()
        has_tracker = True
    except Exception:
        has_tracker = False

    run_id = config_to_id(config)
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress_fh = open(PROGRESS_FILE, "w")
    progress_fh.write(f">>> START [{run_id}] arch={arch}\n")
    progress_fh.flush()
    print(f"\n>>> START [{run_id}] arch={arch}", flush=True)
    _orig_stdout = sys.stdout
    sys.stdout = _Tee(_orig_stdout, progress_fh)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r.start()
    finally:
        sys.stdout = _orig_stdout
        progress_fh.close()
    if has_tracker:
        tracker.stop()
    r.stop()
    elapsed = time.time() - t0
    print(f">>> END   [{run_id}] arch={arch} elapsed={elapsed:.0f}s", flush=True)
    with open(PROGRESS_FILE, "a") as fh:
        fh.write(f">>> END   [{run_id}] arch={arch} elapsed={elapsed:.0f}s\n")

    with open(artifacts_path / "params.json", "w") as f:
        json.dump(config, f, indent=2, default=str)

    n_concepts = len(train_ds_list)
    metrics = compute_metrics(artifacts_path, n_concepts)
    return {
        "dataset": "smd", "model": arch,
        **config,
        **metrics,
        "n_concepts": n_concepts,
        "elapsed_s": round(elapsed, 1),
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="all", choices=ARCHITECTURES + ["all"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int, default=None,
                        help="Run only the first N pending configs (sanity testing)")
    args = parser.parse_args()

    arches = ARCHITECTURES if args.model == "all" else [args.model]
    configs = generate_configs()
    all_runs = [(arch, c) for arch in arches for c in configs]
    total = len(all_runs)

    out_dir = ROOT / "experiments" / "smd"
    out_csv = out_dir / "summary.csv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"SMD Experiment Runner")
    print(f"Architectures: {arches}")
    print(f"Configs per arch: {len(configs)}")
    print(f"Total runs: {total}")
    print(f"Output: {out_csv}\n")

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

    # Resume
    results = []
    existing_ids = set()
    if args.resume and out_csv.exists():
        existing = pd.read_csv(out_csv)
        results = existing.to_dict("records")
        existing_ids = set(
            f"{r['model']}_{config_to_id(r)}" for r in results)
        print(f"Resuming: {len(results)} existing results\n")

    pending = [(arch, c) for arch, c in all_runs
               if f"{arch}_{config_to_id(c)}" not in existing_ids]

    if args.limit is not None:
        pending = pending[:args.limit]

    if len(pending) < total - len(existing_ids):
        print(f"Limited to first {len(pending)} pending runs\n")

    pbar = tqdm(pending, desc="SMD experiments", unit="run",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")
    for arch, config in pbar:
        run_id = config_to_id(config)
        artifacts = out_dir / arch / run_id
        os.makedirs(artifacts, exist_ok=True)
        pbar.set_postfix_str(f"{arch}/{config['regime']}", refresh=True)
        try:
            row = run_single(config, arch, artifacts)
            row["run_id"] = run_id
            results.append(row)
            pbar.set_postfix_str(
                f"{arch}/{config['regime']} BWT={row['bwt']:+.3f} "
                f"AUC={row['final_avg_auc']:.3f}", refresh=True)
            pd.DataFrame(results).to_csv(out_csv, index=False)
        except Exception as e:
            tqdm.write(f"ERROR [{arch}/{run_id}]: {e}")

    pbar.close()
    print(f"\nDone: {len(results)} runs. Results: {out_csv}")


if __name__ == "__main__":
    main()
