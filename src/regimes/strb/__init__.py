import random
from collections import Counter

import torch

from config import SEED
from src.regimes.replay_base import ReplayBaseRegime
from src.utils.reproducibility import ensure_reproducibility


class STRBRegime(ReplayBaseRegime):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.novelty_threshold: float = kwargs.get("novelty_threshold", 0.05)
        self.retention_threshold: float = kwargs.get("retention_threshold", 0.9)
        self._last_novelty_values: list[float] = []
        self._last_retention_values: list[float] = []
        self._last_n_replacements: int = 0
        self._last_n_skipped: int = 0
        self._last_n_candidates: int = 0
        self._last_replacements_by_concept: dict[int, int] = {}
        self._last_retention_by_concept: dict[int, float] = {}
        self._last_buffer_concept_dist: dict[int, int] = {}
        self._last_recency_weight_summary: dict[str, float] = {}

    def update_replay_buffer(self, current_dataset, current_concept_idx: int) -> None:
        self._update_buffer(current_dataset, current_concept_idx)

    def _extra_buffer_stats(self) -> dict:
        import numpy as np
        stats = {
            "n_novel_candidates": self._last_n_candidates,
            "n_replacements": self._last_n_replacements,
            "n_skipped": self._last_n_skipped,
            "replacements_by_concept": str(self._last_replacements_by_concept),
            "retention_by_concept": str({k: round(v, 4) for k, v in self._last_retention_by_concept.items()}),
            "buffer_concept_dist": str(self._last_buffer_concept_dist),
            "recency_weights": str({k: round(v, 4) for k, v in self._last_recency_weight_summary.items()}),
        }
        if self._last_novelty_values:
            nv = self._last_novelty_values
            stats.update(
                avg_novelty=round(float(np.mean(nv)), 4),
                std_novelty=round(float(np.std(nv)), 4),
                min_novelty=round(float(min(nv)), 4),
                max_novelty=round(float(max(nv)), 4),
            )
        if self._last_retention_values:
            rv = self._last_retention_values
            stats.update(
                avg_retention=round(float(np.mean(rv)), 4),
                std_retention=round(float(np.std(rv)), 4),
                min_retention=round(float(min(rv)), 4),
                max_retention=round(float(max(rv)), 4),
            )
        return stats

    def _encode_batch(self, x_batch: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            if self.variational:
                mean, _logvar = self.autoencoder.encoder(x_batch)
                z = mean  # deterministic posterior mean for stable buffer indexing
            else:
                z = self.autoencoder.encoder(x_batch)
        z = z.cpu()
        while z.dim() > 2 and z.shape[1] == 1:
            z = z.squeeze(1)
        return z

    def _update_buffer(self, current_dataset, concept_idx: int) -> None:
        ensure_reproducibility(SEED)

        # STEP 1: Batched encode + retention for existing buffer
        retained_entries: list[dict] = []
        if self.replay_buffer:
            buf_x = torch.stack([e["x"] for e in self.replay_buffer])
            z_buf_new = self._encode_batch(buf_x)
            z_buf_old = torch.stack([e["z_old"] for e in self.replay_buffer]).cpu()

            z_new_n = z_buf_new / z_buf_new.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z_old_n = z_buf_old / z_buf_old.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            retentions = (z_new_n * z_old_n).sum(dim=-1)

            for i, e in enumerate(self.replay_buffer):
                retained_entries.append({
                    "x": e["x"],
                    "z_old": e["z_old"],
                    "z_new": z_buf_new[i],
                    "retention": retentions[i].item(),
                    "concept": e.get("concept", 0),
                })

        # STEP 2: Batched encode for current concept's samples
        n = len(current_dataset)
        new_x_full = torch.stack([current_dataset[i][0] for i in range(n)])
        z_new_full = self._encode_batch(new_x_full)

        # STEP 3: Vectorized novelty per new sample
        if not retained_entries:
            novelty_values = [1.0] * n
        else:
            z_old_retained = torch.stack(
                [e["z_old"] for e in retained_entries]
            ).cpu()
            z_new_n2 = z_new_full / z_new_full.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            z_old_n2 = z_old_retained / z_old_retained.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            sim_matrix = z_new_n2 @ z_old_n2.T
            max_sim = sim_matrix.max(dim=1).values
            novelty_values = (1.0 - max_sim).tolist()

        new_candidates: list[dict] = []
        for i in range(n):
            if novelty_values[i] < self.novelty_threshold:
                continue
            new_candidates.append({
                "x": new_x_full[i].cpu(),
                "z": z_new_full[i],
                "novelty": float(novelty_values[i]),
            })

        self._last_novelty_values = novelty_values
        self._last_retention_values = (
            [e["retention"] for e in retained_entries] if retained_entries else []
        )
        self._last_n_candidates = len(new_candidates)

        if novelty_values:
            print(
                f"[NOVELTY] avg: {sum(novelty_values)/len(novelty_values):.4f} | "
                f"min: {min(novelty_values):.4f} | max: {max(novelty_values):.4f}",
                flush=True,
            )

        # STEP 4 (concept 0): Initialization
        space_left = self.buffer_size - len(self.replay_buffer)
        if concept_idx == 0 and space_left > 0:
            print("[STRB] concept 0: diverse-fill initialization", flush=True)
            z_matrix = torch.stack([e["z"] for e in new_candidates])
            z_norm = z_matrix / z_matrix.norm(dim=1, keepdim=True).clamp(min=1e-8)
            similarity_matrix = torch.matmul(z_norm, z_norm.T)
            dissim_matrix = 1.0 - similarity_matrix

            selected = [0]
            remaining = list(range(1, len(z_matrix)))

            while len(selected) < min(space_left, len(z_matrix)) and remaining:
                sims = dissim_matrix[remaining][:, selected]
                min_sim = torch.min(sims, dim=1).values
                next_idx = remaining[torch.argmax(min_sim).item()]
                selected.append(next_idx)
                remaining.remove(next_idx)

            for idx in selected:
                e = new_candidates[idx]
                self.replay_buffer.append(
                    {"x": e["x"], "z_old": e["z"], "concept": concept_idx}
                )
                retained_entries.append({
                    "x": e["x"], "z_old": e["z"], "z_new": e["z"],
                    "retention": 1.0, "concept": concept_idx,
                })
            # Stats finalisation for concept 0
            self._last_n_replacements = 0
            self._last_n_skipped = 0
            self._last_replacements_by_concept = {}
            self._last_buffer_concept_dist = dict(Counter(
                e.get("concept", 0) for e in self.replay_buffer
            ))
            # Per-concept retention summary (all are 1.0 at concept 0 init)
            self._last_retention_by_concept = {concept_idx: 1.0}
            print(
                f"[STRB] buffer init: size={len(self.replay_buffer)}/{self.buffer_size} "
                f"dist={self._last_buffer_concept_dist}",
                flush=True,
            )
            return

        # ── STEP 5 (concepts 1+): selective replacement with recency weighting ──
        # For each new candidate (passing novelty filter), pick a replaceable
        # buffer slot weighted by recency (admission_concept + 1 — newer = higher
        # weight = more likely replaced); replace if novelty > 1 - retention.
        num_replacements = 0
        already_replaced: dict[int, float] = {}
        replacements_by_concept: dict[int, int] = {}

        retention_vals = [e["retention"] for e in retained_entries]
        candidates = [
            i for i, r in enumerate(retention_vals)
            if r > self.retention_threshold
        ]

        # Recency weight: newer admission = higher weight (replaced first)
        recency_per_idx = {
            i: float(retained_entries[i]["concept"] + 1)
            for i in candidates
        }

        # Per-concept retention stats (for observability)
        per_concept_retention: dict[int, list[float]] = {}
        for entry in retained_entries:
            per_concept_retention.setdefault(entry["concept"], []).append(entry["retention"])
        self._last_retention_by_concept = {
            c: sum(rs) / len(rs) for c, rs in per_concept_retention.items()
        }

        # Recency weight summary (for observability)
        if recency_per_idx:
            ws = list(recency_per_idx.values())
            self._last_recency_weight_summary = {
                "min": float(min(ws)),
                "max": float(max(ws)),
                "mean": float(sum(ws) / len(ws)),
                "n_replaceable_candidates": float(len(ws)),
            }
        else:
            self._last_recency_weight_summary = {}

        def _build_weights():
            ws = [recency_per_idx[i] for i in candidates]
            total = sum(ws)
            return [w / total for w in ws] if total > 0 else None

        HEARTBEAT_EVERY = (
            max(1000, len(new_candidates) // 10) if new_candidates else 1
        )
        n_processed = 0

        for new in new_candidates:
            n_processed += 1
            if n_processed % HEARTBEAT_EVERY == 0:
                print(
                    f"[REPLACE] processed {n_processed}/{len(new_candidates)} "
                    f"candidates, {num_replacements} replacements so far",
                    flush=True,
                )

            if not candidates:
                continue

            weights_list = _build_weights()
            if weights_list is None:
                continue

            tried: set[int] = set()
            while len(tried) < len(candidates):
                replace_idx = random.choices(
                    candidates, weights=weights_list, k=1
                )[0]

                redundant_retention = retention_vals[replace_idx]
                forgetting_score = 1.0 - redundant_retention

                do_replace = False
                if replace_idx in already_replaced:
                    if new["novelty"] > already_replaced[replace_idx]:
                        do_replace = True
                else:
                    if new["novelty"] > forgetting_score:
                        do_replace = True

                if do_replace:
                    old_concept = retained_entries[replace_idx]["concept"]
                    replacements_by_concept[old_concept] = (
                        replacements_by_concept.get(old_concept, 0) + 1
                    )

                    self.replay_buffer[replace_idx] = {
                        "x": new["x"], "z_old": new["z"], "concept": concept_idx,
                    }
                    retained_entries[replace_idx] = {
                        "x": new["x"], "z_old": new["z"], "z_new": new["z"],
                        "retention": 1.0, "concept": concept_idx,
                    }
                    retention_vals[replace_idx] = 1.0
                    # New entry's recency weight = concept_idx + 1 (newest)
                    recency_per_idx[replace_idx] = float(concept_idx + 1)
                    already_replaced[replace_idx] = new["novelty"]
                    num_replacements += 1
                    break
                else:
                    tried.add(replace_idx)

        self._last_n_replacements = num_replacements
        self._last_n_skipped = len(new_candidates) - num_replacements
        self._last_replacements_by_concept = dict(replacements_by_concept)
        self._last_buffer_concept_dist = dict(Counter(
            e.get("concept", 0) for e in self.replay_buffer
        ))

        print(
            f"[STRB] concept={concept_idx} replacements={num_replacements} "
            f"by_concept={replacements_by_concept} "
            f"buffer_dist={self._last_buffer_concept_dist} "
            f"retention_by_concept={ {c: round(v, 3) for c, v in self._last_retention_by_concept.items()} }",
            flush=True,
        )
