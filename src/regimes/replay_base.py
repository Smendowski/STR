"""Base class for all replay-based continual learning regimes."""
import json
from collections import Counter
from math import log

import numpy as np
import pandas as pd
import torch
from torch.utils.data import ConcatDataset, TensorDataset

from src.regimes.base import BaseTrainingRegime


class ReplayBaseRegime(BaseTrainingRegime):
    """Shared logic for STRB, RandomBuffer, and Reservoir regimes.

    Handles: replay buffer storage, building replay datasets with
    oversampling, the concept training loop, and buffer statistics logging.
    Subclasses only need to implement ``_update_buffer(dataset, concept_idx)``.
    """

    def __init__(self, buffer_size: int = 100, replay_ratio: float | None = 0.3,
                 save_buffer_content: bool = False,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.replay_buffer: list[dict] = []
        self.buffer_size = buffer_size
        self.replay_ratio = replay_ratio
        self.buffer_stats: list[dict] = []
        # When True, after each concept we dump the raw buffer contents
        # to df_buffer_content_{concept_idx}.parquet, enabling downstream
        # analysis (covariance det, total variance, pairwise distance,
        # cluster entropy, t-SNE) that needs the actual stored sequences.
        self.save_buffer_content = save_buffer_content

    def start(self) -> None:
        self._ensure_reproducibility()
        print(f"Found {len(self.train_datasets)} concepts in dataset")

        for i in range(len(self.train_datasets)):
            print(f"{self.__class__.__name__}: concept {i + 1}/{len(self.train_datasets)}")

            current_dataset = self.train_datasets[i]
            if len(self.replay_buffer) == 0:
                self.train_dataset = current_dataset
            else:
                self.train_dataset = self._build_replay_dataset(current_dataset)

            self.train()
            self.calculate_threshold()
            self.forward_pass_train_datasets()
            self.forward_pass_test_datasets()
            self._update_buffer(current_dataset, i)
            self._log_buffer_stats(current_dataset, i)
            self.tick()

    def stop(self) -> None:
        super().stop()
        if self.buffer_stats:
            pd.DataFrame(self.buffer_stats).to_parquet(
                self.artifacts_path / "df_buffer_stats.parquet"
            )

    def _update_buffer(self, current_dataset, concept_idx: int) -> None:
        raise NotImplementedError

    def _log_buffer_stats(self, current_dataset, concept_idx: int) -> None:
        """Log buffer statistics after each concept transition."""
        n_buf = len(self.replay_buffer)
        n_concept = len(current_dataset)

        # Concept distribution
        concepts = [e.get("concept", 0) for e in self.replay_buffer]
        dist = dict(Counter(concepts))

        # Shannon entropy of concept distribution
        if n_buf > 0 and len(dist) > 0:
            probs = [count / n_buf for count in dist.values()]
            entropy = -sum(p * log(p) for p in probs if p > 0)
        else:
            entropy = 0.0

        # Effective replay ratio
        replay_ratio_eff = n_buf / (n_buf + n_concept) if (n_buf + n_concept) > 0 else 0.0

        stats = {
            "concept_idx": concept_idx,
            "buffer_size_actual": n_buf,
            "buffer_concept_distribution": json.dumps(dist, sort_keys=True),
            "buffer_concept_entropy": round(entropy, 4),
            "buffer_n_concepts_represented": len(dist),
            "n_samples_in_concept": n_concept,
            "replay_ratio_effective": round(replay_ratio_eff, 4),
        }

        # Subclass-specific stats (novelty, retention, replacements)
        extra = self._extra_buffer_stats()
        stats.update(extra)

        self.buffer_stats.append(stats)

        # Optional: dump raw buffer content for offline profile analysis.
        if self.save_buffer_content and self.replay_buffer:
            self._dump_buffer_content(concept_idx)

    def _dump_buffer_content(self, concept_idx: int) -> None:
        """Write raw buffer contents to df_buffer_content_{concept_idx}.parquet.

        Each row = one buffer slot. Columns:
          slot_idx        position in the buffer
          source_concept  the concept this sample originated from
          seq_shape       (seq_len, n_features) tuple
          x_flat          flattened sequence values (list of floats)
          z_latent        flattened latent code if the entry stores one,
                          else NULL
        """
        records = []
        for slot_idx, entry in enumerate(self.replay_buffer):
            x = entry["x"]
            if hasattr(x, "detach"):
                x_np = x.detach().cpu().numpy()
            else:
                x_np = np.asarray(x)
            record = {
                "concept_idx":    concept_idx,
                "slot_idx":       slot_idx,
                "source_concept": int(entry.get("concept", 0)),
                "seq_shape":      list(x_np.shape),
                "x_flat":         x_np.flatten().astype(float).tolist(),
            }
            # Optional latent (STR stores latent representations).
            z = entry.get("z") or entry.get("latent") or entry.get("mu")
            if z is not None:
                if hasattr(z, "detach"):
                    z_np = z.detach().cpu().numpy()
                else:
                    z_np = np.asarray(z)
                record["z_latent"] = z_np.flatten().astype(float).tolist()
            else:
                record["z_latent"] = None
            records.append(record)
        pd.DataFrame(records).to_parquet(
            self.artifacts_path / f"df_buffer_content_{concept_idx}.parquet"
        )

    def _extra_buffer_stats(self) -> dict:
        """Override in subclasses to add strategy-specific stats."""
        return {}

    def _build_replay_dataset(self, current_dataset):
        buf_ds = self._buffer_to_dataset()
        n_current = len(current_dataset)
        n_buf = len(buf_ds)

        if self.replay_ratio is not None and self.replay_ratio > 0 and n_buf > 0:
            target = self.replay_ratio * n_current / (1.0 - self.replay_ratio)
            repeats = max(1, round(target / n_buf))
            if repeats > 1:
                buf_ds = ConcatDataset([buf_ds] * repeats)
                print(f"[REPLAY] Oversampling {repeats}× "
                      f"({n_buf}→{len(buf_ds)}) ~{self.replay_ratio:.0%} ratio")
        return ConcatDataset([current_dataset, buf_ds])

    def _buffer_to_dataset(self) -> TensorDataset:
        example_ys = self.train_dataset[0][1]
        ys = example_ys.new_zeros((len(self.replay_buffer),) + example_ys.shape)
        xs = torch.stack([entry["x"] for entry in self.replay_buffer])
        return TensorDataset(xs, ys)
