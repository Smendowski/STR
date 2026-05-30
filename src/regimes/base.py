import copy
from pathlib import Path
from typing import Type

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, random_split

from src.utils.data import TemporalConceptDataset
from src.utils.reproducibility import ensure_reproducibility


class BaseTrainingRegime:
    def __init__(
        self,
        autoencoder: torch.nn.Module,
        variational: bool,
        learning_rate: float,
        epochs: int,
        batch_size: int,
        sequence_len: int,
        optimizer_cls: Type[torch.optim.Optimizer],
        criterion_cls: Type[torch.nn.Module],
        criterion_fn,
        train_datasets: list[TemporalConceptDataset],
        test_datasets: list[TemporalConceptDataset],
        seed: int,
        early_stopping_patience: int,
        early_stopping_min_delta: float,
        artifacts_path: Path,
        *args,
        **kwargs,
    ) -> None:
        self.autoencoder = autoencoder
        self.autoencoder_safe_copy = copy.deepcopy(autoencoder)

        self.variational = variational
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_len = sequence_len
        self.optimizer_cls = optimizer_cls
        self.optimizer = self.optimizer_cls(
            self.autoencoder.parameters(), lr=self.learning_rate
        )
        self.criterion_cls = criterion_cls
        self.criterion_fn = criterion_fn
        self.criterion = self.criterion_cls()
        self.train_datasets = train_datasets
        self.test_datasets = test_datasets
        self.seed = seed
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_min_delta = early_stopping_min_delta
        self.artifacts_path = artifacts_path
        self.verbose = kwargs.get("verbose", True)

        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.8,
            patience=2,
            verbose=False,
            min_lr=1e-6,
        )

        self.train_dataset = None
        self.threshold = 0
        self.curr_train_concept = 0
        self.reconstruction_error = None

        # Metadata
        self.training_stats: list[dict] = []
        self.train_concept_stats: list[dict] = []
        self.test_concept_stats: list[dict] = []
        self.threshold_stats: list[dict] = []
        self.classification_stats: list[dict] = []

    def start(self) -> None:
        raise NotImplementedError()

    def stop(self) -> None:
        pd.DataFrame(self.training_stats).to_parquet(
            self.artifacts_path / "df_training_stats.parquet"
        )
        pd.DataFrame(self.train_concept_stats).to_parquet(
            self.artifacts_path / "df_train_concept_stats.parquet"
        )
        pd.DataFrame(self.test_concept_stats).to_parquet(
            self.artifacts_path / "df_test_concept_stats.parquet"
        )
        pd.DataFrame(self.classification_stats).to_parquet(
            self.artifacts_path / "df_classification_stats.parquet"
        )

    def tick(self) -> None:
        self.curr_train_concept += 1

    def train(self) -> None:
        self._ensure_reproducibility()
        self.autoencoder.train()

        best_val_loss = float("inf")
        epochs_without_improvement = 0
        holdout_size: float = int(0.2 * len(self.train_dataset))
        train_size: float = len(self.train_dataset) - holdout_size
        train_subset, holdout_subset = random_split(
            self.train_dataset, [train_size, holdout_size]
        )

        epoch_id: int
        for epoch_id in range(self.epochs):
            # KL-warmup support: losses with a `current_epoch` attribute
            # (e.g., WarmupBetaVariationalMSELoss) get the current epoch
            # so they can anneal β. No-op for losses without this attribute.
            if hasattr(self.criterion, "current_epoch"):
                self.criterion.current_epoch = epoch_id
            epoch_loss: float = 0.0
            train_dataloader = DataLoader(
                train_subset, batch_size=self.batch_size, shuffle=True
            )
            holdout_dataloader = DataLoader(
                holdout_subset, batch_size=self.batch_size, shuffle=False
            )

            n_batches: int = len(train_dataloader)

            for batch_data, _ in train_dataloader:
                self.autoencoder.train()
                batch_data = batch_data.detach().clone()

                self.optimizer.zero_grad()

                if self.variational:
                    reconstruction, mean, logvar = self.autoencoder(batch_data)
                    loss = self.criterion(
                        reconstruction, batch_data, mean, logvar
                    )
                else:
                    reconstruction = self.autoencoder(batch_data)
                    loss = self.criterion(reconstruction, batch_data)

                loss.backward()
                self.optimizer.step()

                with torch.no_grad():
                    self.autoencoder.eval()
                    reconstruction_loss = (
                        self.criterion_fn(
                            reconstruction, batch_data, reduction="none"
                        )
                        .sum()
                        .item()
                    )
                    epoch_loss += reconstruction_loss

            self.calculate_training_stats(epoch_id, epoch_loss, n_batches)
            self.lr_scheduler.step(epoch_loss)

            val_loss: float = 0.0
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
                        self.criterion_fn(
                            reconstruction, batch_data, reduction="none"
                        )
                        .sum()
                        .item()
                    )
                    val_loss += batch_val_loss

            print(
                f"\t -> Epoch {epoch_id + 1}/{self.epochs} train loss: {epoch_loss:.3f} | val loss: {val_loss:.3f}"
            )

            if best_val_loss - val_loss > self.early_stopping_min_delta:
                best_val_loss = val_loss
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= self.early_stopping_patience:
                print(
                    f"\t [EARLY STOPPING] No improvement on validation loss for {self.early_stopping_patience} epochs."
                )
                break

    def calculate_training_stats(
        self, epoch_id: int, epoch_loss: float, n_batches: int
    ) -> None:
        mean_loss_per_batch = epoch_loss / n_batches
        mean_loss_per_sequence = mean_loss_per_batch / self.batch_size
        mean_loss_per_sample = mean_loss_per_sequence / self.sequence_len

        metadata = {
            "current_train_concept": self.curr_train_concept,
            "epoch": epoch_id,
            "total_loss": epoch_loss,
            "mean_loss_per_batch": mean_loss_per_batch,
            "mean_loss_per_sequence": mean_loss_per_sequence,
            "mean_loss_per_sample": mean_loss_per_sample,
        }

        self.training_stats.append(metadata)

    def calculate_threshold(self) -> None:
        self._ensure_reproducibility()

        self.reconstruction_error = None

        with torch.no_grad():
            self.autoencoder.eval()

            train_dataloader = DataLoader(
                self.train_dataset, batch_size=self.batch_size, shuffle=False
            )
            for batch_data, batch_labels in train_dataloader:
                if self.variational:
                    reconstruction, mean, logvar = self.autoencoder(batch_data)
                else:
                    reconstruction = self.autoencoder(batch_data)

                # Sum over features (multivariate-safe): one error per timestep
                _reconstruction_error = (
                    self.criterion_fn(
                        reconstruction, batch_data, reduction="none"
                    )
                    .sum(dim=-1)   # (batch, seq_len)
                    .flatten()     # (batch * seq_len,)
                )

                if self.reconstruction_error is None:
                    self.reconstruction_error = _reconstruction_error
                else:
                    self.reconstruction_error = torch.cat(
                        (self.reconstruction_error, _reconstruction_error),
                        dim=0,
                    )

        self.threshold = np.quantile(self.reconstruction_error, 0.95)

        metadata = {
            "train_concept": self.curr_train_concept,
            "threshold": self.threshold,
        }
        self.threshold_stats.append(metadata)

    def forward_pass_train_datasets(self) -> None:
        self._ensure_reproducibility()

        with torch.no_grad():
            self.autoencoder.eval()

            concept_id: int
            for concept_id, dataset in enumerate(self.train_datasets):
                concept_loss: float = 0.0
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=False
                )
                n_batches: int = len(dataloader)

                for batch_data, batch_labels in dataloader:
                    if self.variational:
                        reconstruction, mean, logvar = self.autoencoder(
                            batch_data
                        )
                    else:
                        reconstruction = self.autoencoder(batch_data)

                    _reconstruction_error = (
                        self.criterion_fn(
                            reconstruction, batch_data, reduction="none"
                        )
                        .sum()
                        .item()
                    )

                    concept_loss += _reconstruction_error

                self.calculate_train_concept_stats(
                    concept_id, concept_loss, n_batches
                )
                position: str = (
                    "FUTURE" if concept_id > self.curr_train_concept else "PAST"
                )
                print(
                    f"\t\t -> {position} train_concept_{concept_id} total reconstruction loss: {concept_loss:.3f}"
                )

    def calculate_train_concept_stats(
        self, concept_id: int, concept_loss: float, n_batches: int
    ) -> None:
        mean_loss_per_batch = concept_loss / n_batches
        mean_loss_per_sequence = mean_loss_per_batch / self.batch_size
        mean_loss_per_sample = mean_loss_per_sequence / self.sequence_len

        metadata = {
            "current_train_concept": self.curr_train_concept,
            "reference_train_concept": concept_id,
            "total_loss": concept_loss,
            "mean_loss_per_batch": mean_loss_per_batch,
            "mean_loss_per_sequence": mean_loss_per_sequence,
            "mean_loss_per_sample": mean_loss_per_sample,
        }

        self.train_concept_stats.append(metadata)

    def forward_pass_test_datasets(self) -> None:
        self._ensure_reproducibility()

        with torch.no_grad():
            self.autoencoder.eval()

            concept_id: int
            for concept_id, dataset in enumerate(self.test_datasets):
                concept_loss: float = 0.0
                dataloader = DataLoader(
                    dataset, batch_size=self.batch_size, shuffle=False
                )
                n_batches: int = len(dataloader)

                container_is_anomaly = torch.Tensor()
                container_labels = torch.Tensor()
                container_reconstruction_error = torch.Tensor()

                for batch_data, batch_labels in dataloader:
                    if self.variational:
                        reconstruction, mean, logvar = self.autoencoder(
                            batch_data
                        )
                    else:
                        reconstruction = self.autoencoder(batch_data)
                    _reconstruction_error = (
                        self.criterion_fn(
                            reconstruction, batch_data, reduction="none"
                        )
                        .sum()
                        .item()
                    )

                    concept_loss += _reconstruction_error

                    # Anomaly score: sum reconstruction error over features so we
                    # get one score per timestep - required for multivariate data
                    # where n_features > 1.  For univariate (n_features=1) the sum
                    # is a no-op.  Shape: (batch, seq_len, n_features) → (batch*seq_len,)
                    unfolded_reconstruction_error = (
                        self.criterion_fn(
                            reconstruction, batch_data, reduction="none"
                        )
                        .sum(dim=-1)   # aggregate across features → (batch, seq_len)
                        .flatten()     # → (batch * seq_len,)
                    )

                    # Labels are per-timestep (not per-feature), so flatten directly.
                    # Shape: (batch, seq_len, 1) → (batch * seq_len,)
                    flat_labels = batch_labels[..., 0].flatten()

                    is_anomaly = (
                        (unfolded_reconstruction_error > self.threshold)
                        .to(torch.int)
                    )
                    container_reconstruction_error = torch.cat(
                        (container_reconstruction_error, unfolded_reconstruction_error),
                        dim=0,
                    )
                    container_is_anomaly = torch.cat(
                        (container_is_anomaly, is_anomaly), dim=0
                    )
                    container_labels = torch.cat(
                        (container_labels, flat_labels), dim=0
                    )

                self.calculate_test_concept_stats(
                    concept_id, concept_loss, n_batches
                )

                # Dataset geometry extraction for per-timestep aggregation.
                # seq_len must come from the *actual dataset* shape, not
                # self.sequence_len, because Block 2/3 sweeps ablate seq_len
                # per config (e.g. sl=6, 8, 10, 16, 24). Using the config
                # default (12) caused reshape errors on non-default configs.
                n_ts = int(dataset.data_4D.shape[0])
                n_windows = int(dataset.data_4D.shape[1])
                seq_len = int(dataset.data_4D.shape[2])
                test_step = int(getattr(dataset, "step", 1))
                self.calculate_anomaly_detection_stats(
                    concept_id,
                    container_reconstruction_error,
                    container_is_anomaly,
                    container_labels,
                    n_ts=n_ts,
                    n_windows=n_windows,
                    seq_len=seq_len,
                    test_step=test_step,
                )
                position: str = (
                    "FUTURE" if concept_id > self.curr_train_concept else "PAST"
                )
                print(
                    f"\t\t -> {position} test_concept_{concept_id} total reconstruction loss: {concept_loss:.3f}"
                )

    def calculate_test_concept_stats(
        self, concept_id: int, concept_loss: float, n_batches: int
    ) -> None:
        mean_loss_per_batch = concept_loss / n_batches
        mean_loss_per_sequence = mean_loss_per_batch / self.batch_size
        mean_loss_per_sample = mean_loss_per_sequence / self.sequence_len

        metadata = {
            "current_train_concept": self.curr_train_concept,
            "reference_test_concept": concept_id,
            "mean_loss_per_batch": mean_loss_per_batch,
            "mean_loss_per_sequence": mean_loss_per_sequence,
            "mean_loss_per_sample": mean_loss_per_sample,
        }

        self.test_concept_stats.append(metadata)

    def calculate_anomaly_detection_stats(
        self,
        concept_id: int,
        container_reconstruction_error: torch.Tensor,
        container_is_anomaly: torch.Tensor,
        container_labels: torch.Tensor,
        n_ts: int | None = None,
        n_windows: int | None = None,
        seq_len: int | None = None,
        test_step: int = 1,
    ) -> None:
        # Convert the flattened containers to numpy for metric computation.
        error = container_reconstruction_error.numpy()
        _ = container_is_anomaly.numpy()
        y_true = container_labels.numpy()

        # Use passed-in seq_len (from actual dataset shape) rather than
        # self.sequence_len, which is the config default and can differ
        # from the actual data sequence length in Block 2/3 ablations.
        s = seq_len if seq_len is not None else self.sequence_len

        # Reshape flat containers back into structured (ts, window, position)
        # `error` and `y_true` arrive flat with length n_ts * n_windows * seq_len.
        # Restoring the (ts, window, position) grid makes per-timestep aggregation straightforward.
        err_3d = error.reshape(n_ts, n_windows, s)  # error at (ts, window, position)
        lbl_3d = y_true.reshape(n_ts, n_windows, s)  # ground-truth label at same

        # Raw-timestep length per time series:
        # last window ends at raw_t = (n_windows-1) * test_step + (s-1)
        # so the number of distinct raw timesteps covered is that + 1.
        L = (n_windows - 1) * test_step + s

        # Global accumulators - one (score, label) pair per raw timestep,
        # across all time series in this test concept. Three aggregation
        # variants, following TadGAN (Geiger et al., IEEE BigData 2020) which
        # uses median; we also report mean (most common) and max (most
        # sensitive) as sensitivity checks.
        agg_scores_mean: list[float] = []
        agg_scores_median: list[float] = []
        agg_scores_max: list[float] = []
        agg_labels: list[float] = []

        for ts in range(n_ts):
            # For each raw timestep in this TS, collect every score that
            # covers it (scores come from every window at every position
            # where w*test_step + p == raw_t). Labels are identical across
            # a raw timestep's scores, so we just remember one.
            scores_per_t: list[list[float]] = [[] for _ in range(L)]
            label_per_t: list[float] = [0.0] * L
            covered: list[bool] = [False] * L

            for w in range(n_windows):
                for p in range(s):
                    raw_t = w * test_step + p
                    scores_per_t[raw_t].append(float(err_3d[ts, w, p]))
                    label_per_t[raw_t] = float(lbl_3d[ts, w, p])
                    covered[raw_t] = True

            # Aggregate each raw timestep's collected scores three ways.
            # `covered` skips any raw_t that received zero scores (only
            # possible if test_step > s, which we don't use, but safe anyway).
            for raw_t in range(L):
                if not covered[raw_t]:
                    continue
                scores = scores_per_t[raw_t]
                agg_scores_mean.append(sum(scores) / len(scores))
                agg_scores_median.append(float(np.median(scores)))
                agg_scores_max.append(max(scores))
                agg_labels.append(label_per_t[raw_t])

        n_valid_points = len(agg_labels)

        # ROC-AUC requires both classes present. On sparse-anomaly datasets
        # (e.g. yahoo-a2) some test concepts have zero anomalies; we return
        # None there and let compute_metrics aggregate via nanmean.
        has_both_classes = len(set(int(l) for l in agg_labels)) >= 2
        if not has_both_classes:
            print(
                f"\t [WARNING] Only one class present in test_concept_{concept_id}. Skipping ROC AUC."
            )
            roc_auc_mean = roc_auc_median = roc_auc_max = None
        else:
            # ROC-AUC: threshold-free ranking quality (standard detection metric).
            roc_auc_mean = roc_auc_score(agg_labels, agg_scores_mean)
            roc_auc_median = roc_auc_score(agg_labels, agg_scores_median)
            roc_auc_max = roc_auc_score(agg_labels, agg_scores_max)

        # Per-point ROC-AUC per (train, test) concept pair, under three
        # aggregations (mean / median / max) of the overlapping-window scores
        # that cover each raw timestep. Primary reported metric = median
        # (TadGAN precedent, outlier-robust); mean and max are sensitivity
        # checks.
        metadata = {
            "current_train_concept": self.curr_train_concept,
            "reference_test_concept": concept_id,
            "roc_auc": roc_auc_median,
            "roc_auc_median": roc_auc_median,
            "roc_auc_mean": roc_auc_mean,
            "roc_auc_max": roc_auc_max,
            "n_valid_points": n_valid_points,
        }

        self.classification_stats.append(metadata)

    def reset_model(self) -> None:
        self.autoencoder = copy.deepcopy(self.autoencoder_safe_copy)
        self.optimizer = self.optimizer_cls(
            self.autoencoder.parameters(), lr=self.learning_rate
        )
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=0.8,
            patience=2,
            verbose=False,
            min_lr=1e-6,
        )

    def _ensure_reproducibility(self) -> None:
        ensure_reproducibility(self.seed)
