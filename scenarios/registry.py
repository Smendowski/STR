from dataclasses import dataclass
from typing import Callable, Type

import torch.nn as nn
import torch.nn.functional as F

from config import (
    BATCH_SIZE,
    BI_WEEKLY_SPLIT_SIZE,
    EARLY_STOPPING_MIN_DELTA,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LEARNING_RATE,
    SEED,
    SEQUENCE_LENGTH,
    STRB_BUFFER_SIZE,
    STRB_NOVELTY_THRESHOLD,
    TEST_SEQUENCE_STEP,
    TRAIN_SEQUENCE_STEP,
    TRAIN_SIZE,
    YAHOO_A1_LABEL_COLUMN,
    YAHOO_A1_VALUE_COLUMN,
    YAHOO_A2_LABEL_COLUMN,
    YAHOO_A2_VALUE_COLUMN,
    YAHOO_A3_LABEL_COLUMN,
    YAHOO_A3_VALUE_COLUMN,
    YAHOO_A4_LABEL_COLUMN,
    YAHOO_A4_VALUE_COLUMN,
)
from scenarios.configurations import GRU_CONFIG, LSTM_CONFIG, TCN_CONFIG
from src.models.autoencoder.loss import (
    VariationalMSELoss,
    WarmupBetaVariationalMSELoss,
)
from src.models.autoencoder.standard.gru import GRUAutoencoder
from src.models.autoencoder.standard.lstm import LSTMAutoencoder
from src.models.autoencoder.standard.tcn import TCNAutoencoder
from src.models.autoencoder.variational.gru import GRUVariationalAutoencoder
from src.models.autoencoder.variational.lstm import LSTMVariationalAutoencoder
from src.models.autoencoder.variational.tcn import TCNVariationalAutoencoder
from src.utils.data import (
    read_yahoo_a1_df,
    read_yahoo_a2_df,
    read_yahoo_a3_df,
    read_yahoo_a4_df,
)


@dataclass(frozen=True)
class DatasetEntry:
    key: str
    read_fn: Callable
    value_column: str
    label_column: str


DATASETS: dict[str, DatasetEntry] = {
    "yahoo-a1": DatasetEntry(
        key="yahoo-a1", read_fn=read_yahoo_a1_df,
        value_column=YAHOO_A1_VALUE_COLUMN, label_column=YAHOO_A1_LABEL_COLUMN,
    ),
    "yahoo-a2": DatasetEntry(
        key="yahoo-a2", read_fn=read_yahoo_a2_df,
        value_column=YAHOO_A2_VALUE_COLUMN, label_column=YAHOO_A2_LABEL_COLUMN,
    ),
    "yahoo-a3": DatasetEntry(
        key="yahoo-a3", read_fn=read_yahoo_a3_df,
        value_column=YAHOO_A3_VALUE_COLUMN, label_column=YAHOO_A3_LABEL_COLUMN,
    ),
    "yahoo-a4": DatasetEntry(
        key="yahoo-a4", read_fn=read_yahoo_a4_df,
        value_column=YAHOO_A4_VALUE_COLUMN, label_column=YAHOO_A4_LABEL_COLUMN,
    ),
}


@dataclass(frozen=True)
class ModelEntry:
    key: str
    model_cls: Type[nn.Module]
    ae_config: object
    variational: bool
    criterion_cls: Type[nn.Module]
    criterion_fn: Callable


KL_WARMUP_EPOCHS = 20
KL_WARMUP_TARGET_BETA = 1.0


def _make_warmup_loss_cls() -> Type[nn.Module]:
    target_beta = KL_WARMUP_TARGET_BETA
    warmup_epochs = KL_WARMUP_EPOCHS

    class _WarmupBeta(WarmupBetaVariationalMSELoss):
        def __init__(self, reduction: str = "mean"):
            super().__init__(reduction=reduction,
                             target_beta=target_beta,
                             warmup_epochs=warmup_epochs)

    _WarmupBeta.__name__ = f"WarmupBetaVAELoss_b{target_beta}_w{warmup_epochs}"
    return _WarmupBeta


_WARMUP_LOSS_CLS = _make_warmup_loss_cls()


MODELS: dict[str, ModelEntry] = {
    "gru": ModelEntry(
        key="gru", model_cls=GRUAutoencoder, ae_config=GRU_CONFIG,
        variational=False, criterion_cls=nn.MSELoss, criterion_fn=F.mse_loss,
    ),
    "lstm": ModelEntry(
        key="lstm", model_cls=LSTMAutoencoder, ae_config=LSTM_CONFIG,
        variational=False, criterion_cls=nn.MSELoss, criterion_fn=F.mse_loss,
    ),
    "tcn": ModelEntry(
        key="tcn", model_cls=TCNAutoencoder, ae_config=TCN_CONFIG,
        variational=False, criterion_cls=nn.MSELoss, criterion_fn=F.mse_loss,
    ),
    "vgru": ModelEntry(
        key="vgru", model_cls=GRUVariationalAutoencoder, ae_config=GRU_CONFIG,
        variational=True, criterion_cls=VariationalMSELoss, criterion_fn=F.mse_loss,
    ),
    "vlstm": ModelEntry(
        key="vlstm", model_cls=LSTMVariationalAutoencoder, ae_config=LSTM_CONFIG,
        variational=True, criterion_cls=VariationalMSELoss, criterion_fn=F.mse_loss,
    ),
    "vtcn": ModelEntry(
        key="vtcn", model_cls=TCNVariationalAutoencoder, ae_config=TCN_CONFIG,
        variational=True, criterion_cls=VariationalMSELoss, criterion_fn=F.mse_loss,
    ),
    "vgru-kl": ModelEntry(
        key="vgru-kl", model_cls=GRUVariationalAutoencoder, ae_config=GRU_CONFIG,
        variational=True, criterion_cls=_WARMUP_LOSS_CLS, criterion_fn=F.mse_loss,
    ),
    "vlstm-kl": ModelEntry(
        key="vlstm-kl", model_cls=LSTMVariationalAutoencoder, ae_config=LSTM_CONFIG,
        variational=True, criterion_cls=_WARMUP_LOSS_CLS, criterion_fn=F.mse_loss,
    ),
    "vtcn-kl": ModelEntry(
        key="vtcn-kl", model_cls=TCNVariationalAutoencoder, ae_config=TCN_CONFIG,
        variational=True, criterion_cls=_WARMUP_LOSS_CLS, criterion_fn=F.mse_loss,
    ),
}


TRAINING_KWARGS = dict(
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    sequence_len=SEQUENCE_LENGTH,
    seed=SEED,
    early_stopping_patience=EARLY_STOPPING_PATIENCE,
    early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
)

CONTINUAL_KWARGS = dict(
    novelty_threshold=STRB_NOVELTY_THRESHOLD,
    buffer_size=STRB_BUFFER_SIZE,
)

DATA_KWARGS = dict(
    split_size=BI_WEEKLY_SPLIT_SIZE,
    sequence_length=SEQUENCE_LENGTH,
    train_size=TRAIN_SIZE,
    train_sequence_step=TRAIN_SEQUENCE_STEP,
    test_sequence_step=TEST_SEQUENCE_STEP,
)
