from typing import Sequence

import torch.nn as nn

from config import N_FEATURES, SEQUENCE_LENGTH
from src.models.autoencoder.config import (
    ActivationLayerConfig,
    AutoencoderConfig,
    Conv1dLayerConfig,
    ConvTranspose1dLayerConfig,
    DecoderConfig,
    DropoutLayerConfig,
    EncoderConfig,
    GRULayerConfig,
    LSTMLayerConfig,
)

LSTM_CONFIG: AutoencoderConfig = AutoencoderConfig(
    seq_len=SEQUENCE_LENGTH,
    encoder=EncoderConfig(
        layers=[
            LSTMLayerConfig(
                kwargs={
                    "input_size": N_FEATURES,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "batch_first": True,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
            LSTMLayerConfig(
                kwargs={
                    "input_size": 8,
                    "hidden_size": 4,
                    "num_layers": 1,
                    "batch_first": True,
                }
            ),
        ]
    ),
    decoder=DecoderConfig(
        layers=[
            LSTMLayerConfig(
                kwargs={
                    "input_size": 4,
                    "hidden_size": 8,
                    "num_layers": 1,
                    "batch_first": True,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
            LSTMLayerConfig(
                kwargs={
                    "input_size": 8,
                    "hidden_size": N_FEATURES,
                    "num_layers": 1,
                    "batch_first": True,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
        ]
    ),
)

GRU_CONFIG = AutoencoderConfig(
    seq_len=SEQUENCE_LENGTH,
    encoder=EncoderConfig(
        layers=[
            GRULayerConfig(kwargs={
                "input_size": N_FEATURES,
                "hidden_size": 8,
                "num_layers": 1,
                "batch_first": True,
            }),
            ActivationLayerConfig(cls=nn.Tanh),
            GRULayerConfig(kwargs={
                "input_size": 8,
                "hidden_size": 4,
                "num_layers": 1,
                "batch_first": True,
            }),
        ]
    ),
    decoder=DecoderConfig(
        layers=[
            GRULayerConfig(kwargs={
                "input_size": 4,
                "hidden_size": 8,
                "num_layers": 1,
                "batch_first": True,
            }),
            ActivationLayerConfig(cls=nn.Tanh),
            GRULayerConfig(kwargs={
                "input_size": 8,
                "hidden_size": N_FEATURES,
                "num_layers": 1,
                "batch_first": True,
            }),
            ActivationLayerConfig(cls=nn.Tanh),
        ]
    )
)

TCN_CONFIG: AutoencoderConfig = AutoencoderConfig(
    seq_len=SEQUENCE_LENGTH,
    encoder=EncoderConfig(
        layers=[
            Conv1dLayerConfig(
                kwargs={
                    "in_channels": N_FEATURES,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
            Conv1dLayerConfig(
                kwargs={
                    "in_channels": 8,
                    "out_channels": 4,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
        ]
    ),
    decoder=DecoderConfig(
        layers=[
            ConvTranspose1dLayerConfig(
                kwargs={
                    "in_channels": 4,
                    "out_channels": 8,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
            ConvTranspose1dLayerConfig(
                kwargs={
                    "in_channels": 8,
                    "out_channels": N_FEATURES,
                    "kernel_size": 3,
                    "stride": 1,
                    "padding": 1,
                    "dilation": 1,
                }
            ),
            ActivationLayerConfig(cls=nn.Tanh),
        ]
    ),
)

__all__: Sequence[str] = ["LSTM_CONFIG", "GRU_CONFIG", "TCN_CONFIG"]
