"""Single-Task Experts: one independent model per concept (oracle baseline)."""
import copy

from src.regimes.base import BaseTrainingRegime


class SingleTaskExpertsRegime(BaseTrainingRegime):
    def start(self) -> None:
        self._ensure_reproducibility()

        for i in range(len(self.train_datasets)):
            print(f"SingleTaskExpertsRegime: concept {i + 1}/{len(self.train_datasets)}")

            # Reset to a fresh model for each concept
            self.reset_model()

            self.train_dataset = self.train_datasets[i]
            self.train()
            self.calculate_threshold()
            self.forward_pass_train_datasets()
            self.forward_pass_test_datasets()
            self.tick()
