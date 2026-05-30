import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader

from src.regimes.base import BaseTrainingRegime


class CumulativeTrainingRegime(BaseTrainingRegime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        self._ensure_reproducibility()
        print(f"Found {len(self.train_datasets)} concepts in dataset")

        for i in range(len(self.train_datasets)):
            print(f"Cumulative training on concepts 1 to {i + 1}")

            # Reinitialize model from scratch
            self.reset_model()

            # Combine datasets
            self.train_dataset = torch.utils.data.ConcatDataset(
                self.train_datasets[: i + 1]
            )

            self.train()
            self.calculate_threshold()
            self.forward_pass_train_datasets()
            self.forward_pass_test_datasets()
            self.tick()
