import numpy as np
import torch


def get_4D_shape(
    data: torch.Tensor, sequence_length: int, step: int = 1
) -> torch.Tensor:
    """
    Takes input tensor with 2D shape: (n_rows, n_time_series).
    Returns tensor with 4D shape: (n_time_series, n_sequences, sequence_length, 1).
    """
    data = data.unsqueeze(-1)

    N = data.size(0) - sequence_length + 1

    overlapping_sequences = torch.stack(
        [data[i : i + sequence_length] for i in range(0, N, step)]
    )

    overlapping_sequences = overlapping_sequences.permute(2, 0, 1, 3)

    return overlapping_sequences


def coerce_types(data: torch.Tensor | np.ndarray) -> torch.Tensor:
    return data if type(data) is torch.Tensor else torch.from_numpy(data)
