import numpy as np
import torch
from scipy.spatial.distance import mahalanobis
from torch.utils.data import DataLoader

from src.regimes.base import BaseTrainingRegime


class NaiveTrainingRegime(BaseTrainingRegime):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def start(self) -> None:
        self._ensure_reproducibility()

        for i in range(len(self.train_datasets)):
            print(f"Training on concept {i + 1}/{len(self.train_datasets)}")
            # Naive training regime means fine-tuning existing model on recent data
            self.train_dataset = self.train_datasets[i]

            self.train()
            self.calculate_threshold()
            self.forward_pass_train_datasets()
            self.forward_pass_test_datasets()
            self.tick()
