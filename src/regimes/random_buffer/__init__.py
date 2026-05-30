import random

from config import SEED
from src.regimes.replay_base import ReplayBaseRegime
from src.utils.reproducibility import ensure_reproducibility


class RandomBufferRegime(ReplayBaseRegime):
    def _update_buffer(self, current_dataset, concept_idx: int) -> None:
        ensure_reproducibility(SEED)

        for i in range(len(current_dataset)):
            x, _ = current_dataset[i]
            x = x.cpu()

            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append({"x": x, "concept": concept_idx})
            else:
                idx = random.randint(0, self.buffer_size - 1)
                self.replay_buffer[idx] = {"x": x, "concept": concept_idx}

        print(f"[RANDOM BUFFER] size={len(self.replay_buffer)}")
