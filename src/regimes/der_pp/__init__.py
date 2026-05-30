import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from config import SEED
from src.regimes.replay_base import ReplayBaseRegime
from src.utils.reproducibility import ensure_reproducibility


class DERPlusPlusRegime(ReplayBaseRegime):
    """Dark Experience Replay++ (Buzzega et al. 2020, NeurIPS)."""

    def __init__(self, alpha: float = 0.5, beta: float = 0.5,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self._n_seen = 0  # for reservoir sampling

    def _update_buffer(self, current_dataset, concept_idx: int) -> None:
        """Reservoir-style admission with stored reconstruction outputs."""
        ensure_reproducibility(SEED)
        n_admitted = 0

        with torch.no_grad():
            for i in range(len(current_dataset)):
                x, _ = current_dataset[i]
                x_in = x.unsqueeze(0)

                # Compute and store reconstruction at admission time
                if self.variational:
                    x_hat, _, _ = self.autoencoder(x_in)
                else:
                    x_hat = self.autoencoder(x_in)
                x_hat_stored = x_hat.squeeze(0).cpu()

                self._n_seen += 1
                if len(self.replay_buffer) < self.buffer_size:
                    self.replay_buffer.append({
                        "x": x.cpu(),
                        "x_hat_old": x_hat_stored,
                        "concept": concept_idx,
                    })
                    n_admitted += 1
                else:
                    # Reservoir sampling: replace with prob bu/n_seen
                    j = random.randint(0, self._n_seen - 1)
                    if j < self.buffer_size:
                        self.replay_buffer[j] = {
                            "x": x.cpu(),
                            "x_hat_old": x_hat_stored,
                            "concept": concept_idx,
                        }
                        n_admitted += 1

        self._last_n_admitted = n_admitted
        from collections import Counter
        self._last_buffer_concept_dist = dict(Counter(
            e.get("concept", 0) for e in self.replay_buffer
        ))
        print(
            f"[DER++] concept={concept_idx} admitted={n_admitted} "
            f"buffer_dist={self._last_buffer_concept_dist}",
            flush=True,
        )

    def _extra_buffer_stats(self) -> dict:
        return {
            "n_admitted": getattr(self, "_last_n_admitted", 0),
            "buffer_concept_dist": str(getattr(self, "_last_buffer_concept_dist", {})),
            "alpha": self.alpha,
            "beta": self.beta,
        }

    def train(self) -> None:
        self._ensure_reproducibility()
        self.autoencoder.train()

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        holdout_size = int(0.2 * len(self.train_dataset))
        train_size = len(self.train_dataset) - holdout_size
        train_subset, holdout_subset = random_split(
            self.train_dataset, [train_size, holdout_size]
        )

        # Pre-stack buffer samples for distillation (if any)
        has_buffer = len(self.replay_buffer) > 0
        if has_buffer:
            buf_x = torch.stack([e["x"] for e in self.replay_buffer])
            buf_x_hat_old = torch.stack([e["x_hat_old"] for e in self.replay_buffer])

        for epoch_id in range(self.epochs):
            # KL warmup support (matches base regime)
            if hasattr(self.criterion, "current_epoch"):
                self.criterion.current_epoch = epoch_id

            epoch_loss = 0.0
            train_dataloader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True
            )
            holdout_dataloader = DataLoader(
                holdout_subset, batch_size=self.batch_size, shuffle=False
            )
            n_batches = len(train_dataloader)

            for batch_data, _ in train_dataloader:
                self.autoencoder.train()
                batch_data = batch_data.detach().clone()
                self.optimizer.zero_grad()

                if self.variational:
                    reconstruction, mean, logvar = self.autoencoder(batch_data)
                    task_loss = self.criterion(reconstruction, batch_data, mean, logvar)
                else:
                    reconstruction = self.autoencoder(batch_data)
                    task_loss = self.criterion(reconstruction, batch_data)

                # DER++ distillation loss (only when buffer has entries)
                if has_buffer:
                    # Sample a small minibatch from buffer for distillation
                    bs = min(self.batch_size, len(self.replay_buffer))
                    idx = torch.randperm(len(self.replay_buffer))[:bs]
                    batch_buf_x = buf_x[idx]
                    batch_buf_x_hat_old = buf_x_hat_old[idx]

                    if self.variational:
                        buf_recon_now, _, _ = self.autoencoder(batch_buf_x)
                    else:
                        buf_recon_now = self.autoencoder(batch_buf_x)

                    # alpha * replay reconstruction loss
                    if self.variational:
                        replay_recon_loss = nn.functional.mse_loss(
                            buf_recon_now, batch_buf_x
                        )
                    else:
                        replay_recon_loss = self.criterion_fn(
                            buf_recon_now, batch_buf_x
                        )
                    # beta * distillation MSE against stored x_hat_old
                    distill_loss = nn.functional.mse_loss(
                        buf_recon_now, batch_buf_x_hat_old
                    )

                    loss = task_loss + self.alpha * replay_recon_loss + self.beta * distill_loss
                else:
                    loss = task_loss

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.autoencoder.eval()
                    reconstruction_loss = (
                        self.criterion_fn(reconstruction, batch_data, reduction="none")
                        .sum().item()
                    )
                    epoch_loss += reconstruction_loss

            self.calculate_training_stats(epoch_id, epoch_loss, n_batches)
            self.lr_scheduler.step(epoch_loss)

            val_loss = 0.0
            self.autoencoder.eval()
            with torch.no_grad():
                for batch_data, _ in holdout_dataloader:
                    batch_data = batch_data.detach().clone()
                    reconstruction = (
                        self.autoencoder(batch_data)
                        if not self.variational
                        else self.autoencoder(batch_data)[0]
                    )
                    batch_val_loss = (
                        self.criterion_fn(reconstruction, batch_data, reduction="none")
                        .sum().item()
                    )
                    val_loss += batch_val_loss

            print(
                f"\t -> Epoch {epoch_id + 1}/{self.epochs} "
                f"train loss: {epoch_loss:.3f} | val loss: {val_loss:.3f}"
            )

            if best_val_loss - val_loss > self.early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"\t [EARLY STOPPING] No improvement on validation loss "
                    f"for {self.early_stopping_patience} epochs."
                )
                break
