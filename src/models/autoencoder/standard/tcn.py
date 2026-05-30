import torch
import torch.nn as nn

from src.models.autoencoder.builder import build
from src.models.autoencoder.config import AutoencoderConfig


class TCNEncoder(nn.Module):
    def __init__(self, encoder: nn.ModuleList, seq_len: int) -> None:
        super(TCNEncoder, self).__init__()
        self.encoder: nn.ModuleList = encoder
        self.seq_len = seq_len

        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, seq_len, in_channels) -> (batch_size, in_channels, seq_len)
        x = x.permute(0, 2, 1)

        for layer in self.encoder:
            x = layer(x)

        x = self.pool(x)

        # (batch_size, 1, out_channels)
        return x.permute(0, 2, 1)


class TCNDecoder(nn.Module):
    def __init__(self, decoder: nn.ModuleList, seq_len: int) -> None:
        super(TCNDecoder, self).__init__()
        self.seq_len = seq_len

        for layer in decoder:
            if isinstance(layer, nn.modules.ConvTranspose1d):
                self.linear = nn.Linear(
                    layer.in_channels, layer.in_channels * self.seq_len
                )
                break

        self.decoder: nn.ModuleList = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (batch_size, 1, out_channels) -> (batch_size, out_channels, 1)
        x = x.permute(0, 2, 1)

        # (batch_size, out_channels, 1) -> (batch_size, out_channels)
        x = x.squeeze(-1)
        batch_size, out_channels = x.shape

        # (batch_size, out_channels) -> (batch_size, out_channels * seq_len)
        x = self.linear(x)

        # (batch_size, out_channels * seq_len) -> (batch_size, out_channels, seq_len)
        x = x.view(batch_size, -1, self.seq_len)

        for layer in self.decoder:
            x = layer(x)

        return x.permute(0, 2, 1)


class TCNAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, device: str = "cpu") -> None:
        super(TCNAutoencoder, self).__init__()
        _encoder, _decoder = build(config)
        self.seq_len = config.seq_len

        self.encoder = TCNEncoder(_encoder, seq_len=self.seq_len)
        self.decoder = TCNDecoder(_decoder, seq_len=self.seq_len)
        self._init_weights()

        self.variational = False
        self.device = device

    def _init_weights(self) -> None:
        self.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity="tanh")
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
