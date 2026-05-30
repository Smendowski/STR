import torch
import torch.nn as nn

from src.models.autoencoder.builder import build
from src.models.autoencoder.config import AutoencoderConfig


class GRUVariationalEncoder(nn.Module):
    def __init__(self, encoder: nn.ModuleList, seq_len: int) -> None:
        super(GRUVariationalEncoder, self).__init__()
        self.encoder: nn.ModuleList = encoder
        self.seq_len = seq_len

        for layer in reversed(self.encoder):
            if isinstance(layer, nn.modules.GRU):
                self.mean_layer = nn.Linear(
                    layer.hidden_size, layer.hidden_size
                )
                self.logvar_layer = nn.Linear(
                    layer.hidden_size, layer.hidden_size
                )
                break

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        hidden_size, hidden_state = None, None
        num_layers = None

        for layer in self.encoder:
            if isinstance(layer, nn.GRU):
                output, hidden_state = layer(x)
                hidden_size = layer.hidden_size
                num_layers = layer.num_layers
                x = output
            else:
                x = layer(x)

        if num_layers > 1:
            hidden_state = hidden_state[-1]  # (num_layers, batch, h) → (batch, h)

        hidden_state = hidden_state.reshape((batch_size, 1, hidden_size))

        mean = self.mean_layer(hidden_state)
        logvar = self.logvar_layer(hidden_state)

        return mean, logvar


class GRUVariationalDecoder(nn.Module):
    def __init__(self, decoder: nn.ModuleList, seq_len: int) -> None:
        super(GRUVariationalDecoder, self).__init__()
        self.seq_len = seq_len

        for layer in decoder:
            if isinstance(layer, nn.modules.GRU):
                self.linear = nn.Linear(
                    layer.input_size, layer.input_size * self.seq_len
                )
                break

        self.decoder: nn.ModuleList = decoder

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = x.squeeze(-1)
        batch_size, hidden_size = x.shape

        # (batch_size, hidden_size) -> (batch_size, hidden_size * seq_len)
        x = self.linear(x)

        # (batch_size, hidden_size * seq_len) -> (batch_size, seq_len, hidden_size)
        x = x.view(batch_size, self.seq_len, -1)

        for layer in self.decoder:
            if isinstance(layer, nn.GRU):
                output, hidden_state = layer(x)
                hidden_size = layer.hidden_size
                x = output
            else:
                x = layer(x)

        return x.reshape((batch_size, self.seq_len, hidden_size))


class GRUVariationalAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, device: str = "cpu") -> None:
        super(GRUVariationalAutoencoder, self).__init__()
        _encoder, _decoder = build(config)
        self.seq_len = config.seq_len

        self.encoder = GRUVariationalEncoder(_encoder, seq_len=self.seq_len)
        self.decoder = GRUVariationalDecoder(_decoder, seq_len=self.seq_len)
        self._init_weights()

        self.variational = True
        self.device = device

    def _init_weights(self) -> None:
        self.apply(init_weights)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, var = self.encoder(x)
        x = self.reparametrize(mean, var)
        x = self.decoder(x)
        return x, mean, var

    @staticmethod
    def reparametrize(mean, var):
        epsilon = torch.randn_like(var)
        z = mean + torch.exp(0.5 * var) * epsilon
        return z


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity="tanh")
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
