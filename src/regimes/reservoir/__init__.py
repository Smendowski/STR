import random

from config import SEED
from src.regimes.replay_base import ReplayBaseRegime
from src.utils.reproducibility import ensure_reproducibility


class ReservoirBufferRegime(ReplayBaseRegime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._total_samples_seen: int = 0

    def _update_buffer(self, current_dataset, concept_idx: int) -> None:
        ensure_reproducibility(SEED)

        for i in range(len(current_dataset)):
            x, _ = current_dataset[i]
            x = x.cpu()
            self._total_samples_seen += 1

            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append({"x": x, "concept": concept_idx})
            else:
                # Algorithm R: replace with probability k/n
                j = random.randint(0, self._total_samples_seen - 1)
                if j < self.buffer_size:
                    self.replay_buffer[j] = {"x": x, "concept": concept_idx}

        print(f"[RESERVOIR] size={len(self.replay_buffer)}, "
              f"total_seen={self._total_samples_seen}")
