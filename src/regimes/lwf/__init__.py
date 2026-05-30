import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.regimes.base import BaseTrainingRegime


class LwFRegime(BaseTrainingRegime):
    def __init__(self, alpha: float = 1.0, temperature: float = 2.0,
                 *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.temperature = temperature
        self._teacher = None

    def start(self) -> None:
        self._ensure_reproducibility()

        for i in range(len(self.train_datasets)):
            print(f"LwFRegime: concept {i + 1}/{len(self.train_datasets)} "
                  f"(alpha={self.alpha}, temp={self.temperature})")

            self.train_dataset = self.train_datasets[i]
            self._current_concept = i

            # Snapshot teacher before training (concepts 1+)
            if i > 0:
                self._teacher = copy.deepcopy(self.autoencoder)
                self._teacher.eval()
                for p in self._teacher.parameters():
                    p.requires_grad = False

            self.train()
            self.calculate_threshold()
            self.forward_pass_train_datasets()
            self.forward_pass_test_datasets()
            self.tick()

    def train(self) -> None:
        self._ensure_reproducibility()
        self.autoencoder.train()

        best_val_loss = float("inf")
        epochs_without_improvement = 0

        from torch.utils.data import random_split
        holdout_size = int(0.2 * len(self.train_dataset))
        train_size = len(self.train_dataset) - holdout_size
        train_subset, holdout_subset = random_split(
            self.train_dataset, [train_size, holdout_size]
        )

        for epoch_id in range(self.epochs):
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

                # Distillation loss (concepts 1+)
                if self._teacher is not None:
                    with torch.no_grad():
                        if self.variational:
                            teacher_out = self._teacher(batch_data)[0]
                        else:
                            teacher_out = self._teacher(batch_data)

                    # Soft distillation: compare scaled outputs
                    distill_loss = nn.functional.mse_loss(
                        reconstruction / self.temperature,
                        teacher_out / self.temperature,
                    )
                    loss = task_loss + self.alpha * distill_loss
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
