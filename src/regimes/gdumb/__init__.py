import random
from collections import Counter

from config import SEED
from src.regimes.replay_base import ReplayBaseRegime
from src.utils.reproducibility import ensure_reproducibility


class GDumbRegime(ReplayBaseRegime):
    def _update_buffer(self, current_dataset, concept_idx: int) -> None:
        ensure_reproducibility(SEED)

        n_admitted = 0
        n_evicted_by_concept: dict[int, int] = {}

        for i in range(len(current_dataset)):
            x, _ = current_dataset[i]
            x = x.cpu()

            if len(self.replay_buffer) < self.buffer_size:
                # Buffer not full — admit directly
                self.replay_buffer.append({"x": x, "concept": concept_idx})
                n_admitted += 1
            else:
                # Buffer full — find the concept with the most entries (largest)
                concept_counts = Counter(
                    e.get("concept", 0) for e in self.replay_buffer
                )
                largest_concept = max(concept_counts, key=concept_counts.get)
                # Pick a random entry from the largest concept
                largest_indices = [
                    j for j, e in enumerate(self.replay_buffer)
                    if e.get("concept", 0) == largest_concept
                ]
                evict_idx = random.choice(largest_indices)
                self.replay_buffer[evict_idx] = {
                    "x": x, "concept": concept_idx,
                }
                n_admitted += 1
                n_evicted_by_concept[largest_concept] = (
                    n_evicted_by_concept.get(largest_concept, 0) + 1
                )

        # Stats for observability
        self._last_n_admitted = n_admitted
        self._last_n_evicted_by_concept = dict(n_evicted_by_concept)
        self._last_buffer_concept_dist = dict(Counter(
            e.get("concept", 0) for e in self.replay_buffer
        ))

        print(
            f"[GDUMB] concept={concept_idx} admitted={n_admitted} "
            f"evicted_by_concept={n_evicted_by_concept} "
            f"buffer_dist={self._last_buffer_concept_dist}",
            flush=True,
        )

    def _extra_buffer_stats(self) -> dict:
        return {
            "n_admitted": getattr(self, "_last_n_admitted", 0),
            "evicted_by_concept": str(getattr(self, "_last_n_evicted_by_concept", {})),
            "buffer_concept_dist": str(getattr(self, "_last_buffer_concept_dist", {})),
        }
