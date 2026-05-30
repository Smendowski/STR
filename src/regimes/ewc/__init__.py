"""EWC: Elastic Weight Consolidation - regularisation-based CL strategy."""
import copy

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.regimes.base import BaseTrainingRegime


class EWCRegime(BaseTrainingRegime):
    def __init__(self, lambda_ewc: float = 1000.0, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.lambda_ewc = lambda_ewc
        self._fisher: dict[str, torch.Tensor] = {}
        self._optimal_params: dict[str, torch.Tensor] = {}

    def start(self) -> None:
        self._ensure_reproducibility()

        for i in range(len(self.train_datasets)):
            print(f"EWCRegime: concept {i + 1}/{len(self.train_datasets)} "
                  f"(lambda={self.lambda_ewc})")

            self.train_dataset = self.train_datasets[i]

            # Store reference to standard train() for override
            self._current_concept = i
            self.train()
            self.calculate_threshold()
            self.forward_pass_train_datasets()
            self.forward_pass_test_datasets()

            # After training, compute Fisher and store optimal params
            self._update_fisher()
            self._store_optimal_params()
            self.tick()

    def train(self) -> None:
        """Override base train() to add EWC penalty to loss."""
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

                # EWC penalty
                ewc_loss = self._ewc_penalty()
                loss = task_loss + ewc_loss

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

    def _ewc_penalty(self) -> torch.Tensor:
        """Compute EWC penalty: sum of Fisher-weighted squared parameter changes."""
        if not self._fisher:
            return torch.tensor(0.0)

        penalty = torch.tensor(0.0)
        for name, param in self.autoencoder.named_parameters():
            if name in self._fisher:
                penalty += (
                    self._fisher[name] * (param - self._optimal_params[name]) ** 2
                ).sum()

        return (self.lambda_ewc / 2.0) * penalty

    def _update_fisher(self) -> None:
        """Compute diagonal Fisher Information Matrix from current concept data."""
        self.autoencoder.eval()

        # Accumulate gradients squared
        fisher = {name: torch.zeros_like(param)
                  for name, param in self.autoencoder.named_parameters()
                  if param.requires_grad}

        dataloader = DataLoader(
            self.train_dataset, batch_size=self.batch_size, shuffle=False
        )
        n_samples = 0

        for batch_data, _ in dataloader:
            self.autoencoder.zero_grad()

            if self.variational:
                reconstruction, mean, logvar = self.autoencoder(batch_data)
                loss = self.criterion(reconstruction, batch_data, mean, logvar)
            else:
                reconstruction = self.autoencoder(batch_data)
                loss = self.criterion(reconstruction, batch_data)

            loss.backward()

            for name, param in self.autoencoder.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += param.grad.data ** 2

            n_samples += batch_data.size(0)

        # Average over samples
        for name in fisher:
            fisher[name] /= max(n_samples, 1)

        # Accumulate with previous Fisher (multi-concept)
        if self._fisher:
            for name in fisher:
                self._fisher[name] = self._fisher[name] + fisher[name]
        else:
            self._fisher = fisher

        print(f"[EWC] Fisher computed, trace={sum(f.sum().item() for f in self._fisher.values()):.4f}")

    def _store_optimal_params(self) -> None:
        """Store current parameters as the optimal reference for EWC penalty."""
        self._optimal_params = {
            name: param.data.clone()
            for name, param in self.autoencoder.named_parameters()
            if param.requires_grad
        }
