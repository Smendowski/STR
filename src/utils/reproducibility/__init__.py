import random
from collections.abc import Sequence

import numpy as np
import torch


def ensure_reproducibility(seed: int) -> None:
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)

    torch.use_deterministic_algorithms(mode=True)


__all__: Sequence[str] = [ensure_reproducibility]
