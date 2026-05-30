import torch
import torch.nn as nn

from src.models.autoencoder.builder import build
from src.models.autoencoder.config import AutoencoderConfig


class GRUEncoder(nn.Module):
    def __init__(self, encoder: nn.ModuleList, seq_len: int) -> None:
        super(GRUEncoder, self).__init__()
        self.encoder: nn.ModuleList = encoder
        self.seq_len = seq_len

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        hidden_state = None

        for layer in self.encoder:
            if isinstance(layer, nn.GRU):
                # output: (batch, seq, hidden)
                # hidden: (num_layers, batch, hidden)
                output, hidden_state = layer(x)
                x = output
            else:
                x = layer(x)

        # Return last-layer hidden state: (batch, hidden)
        latent = hidden_state[-1]

        return latent


class GRUDecoder(nn.Module):
    def __init__(self, decoder: nn.ModuleList, seq_len: int):
        super().__init__()
        self.decoder = decoder
        self.seq_len = seq_len

        # Find input_size of the FIRST GRU layer
        for layer in decoder:
            if isinstance(layer, nn.GRU):
                self.input_size = layer.input_size
                self.hidden_size = layer.hidden_size
                break

        self.hidden_proj = nn.Linear(self.input_size, self.hidden_size)

    def forward(self, latent: torch.Tensor):
        batch = latent.size(0)

        # Feed the latent at every decoder timestep instead of zeros.
        # The latent already has shape (batch, input_size) - exactly what the
        # first decoder GRU expects as input - so expand directly.
        # hidden_proj is still used for the initial hidden state (different size).
        # This gives weight_ih_l0 real gradients; with zero input those weights
        # were permanently dead (confirmed in tests).
        x = latent.unsqueeze(1).expand(-1, self.seq_len, -1)  # (batch, seq_len, input_size)
        initial_hidden = self.hidden_proj(latent).unsqueeze(0)  # (1, batch, hidden_size)

        first_gru = True
        for layer in self.decoder:
            if isinstance(layer, nn.GRU):
                if first_gru:
                    x, _ = layer(x, initial_hidden)
                    first_gru = False
                else:
                    x, _ = layer(x)
            else:
                x = layer(x)

        return x


class GRUAutoencoder(nn.Module):
    def __init__(self, config: AutoencoderConfig, device: str = "cpu") -> None:
        super(GRUAutoencoder, self).__init__()
        _encoder, _decoder = build(config)
        self.seq_len = config.seq_len

        self.encoder = GRUEncoder(_encoder, seq_len=self.seq_len)
        self.decoder = GRUDecoder(_decoder, seq_len=self.seq_len)
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
    if isinstance(m, (nn.GRU, nn.GRUCell)):
        for name, param in m.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param.data, nonlinearity="tanh")
            elif "bias" in name:
                nn.init.constant_(param.data, 0)
