import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
from sklearn.preprocessing import MinMaxScaler, PowerTransformer

from .dataframe import (
    get_df_data,
    get_df_labels,
    get_df_without_anomalies,
    split_df,
)
from .tensor import coerce_types, get_4D_shape


@dataclass
class TemporalConcept:
    name: str
    data: torch.Tensor
    labels: torch.Tensor
    timeline: list[datetime] = None


class TemporalConceptDataset(data.Dataset):
    def __init__(self, concept: TemporalConcept) -> None:
        self.concept = concept
        self.data_4D = coerce_types(self.concept.data)
        self.labels_4D = coerce_types(self.concept.labels)

        # A: Number of time series in the dataset
        # B: Number of sequences
        # C: Time series sequence length
        # D: dimensionality (number of features) of time VM-specific time series in the dataset
        A, B, C, D = self.data_4D.shape

        self.data_3D = self.data_4D.reshape(A * B, C, D)
        self.labels_3D = self.labels_4D.reshape(A * B, C, D)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.data_3D[index][:]
        labels = self.labels_3D[index][:]

        return item, labels

    def __len__(self) -> int:
        return self.data_3D.size(0)


def get_temporal_concept_datasets(
    df: pd.DataFrame,
    value_column: str,
    label_column: str,
    split_size: int,
    sequence_length: int,
    train_size: float,
    train_sequence_step: int,
    test_sequence_step: int,
) -> tuple[list[TemporalConceptDataset], list[TemporalConceptDataset]]:
    train_concept_datasets: list[TemporalConceptDataset] = []
    test_concept_datasets: list[TemporalConceptDataset] = []

    scaler = MinMaxScaler()
    pt = PowerTransformer(method="yeo-johnson")

    split: pd.DataFrame
    for i, split in enumerate(split_df(df, split_size)):
        split = split.reset_index(drop=True)
        sequences = len(split) // sequence_length
        train_sequences = math.floor(train_size * sequences)

        """
        To follow semi-supervised approach the training data should not contain outliers.
        """
        df_train: pd.DataFrame = get_df_without_anomalies(
            df=split.loc[: train_sequences * sequence_length],
            value_column=value_column,
            label_column=label_column,
        ).bfill()
        df_train_data = get_df_data(df_train, value_column)
        df_train_labels = get_df_labels(df_train, label_column)

        df_test: pd.DataFrame = split.loc[train_sequences * sequence_length :]
        df_test_data = get_df_data(df_test, value_column)
        df_test_labels = get_df_labels(df_test, label_column)

        if i == 0:
            train_data_scaled = scaler.fit_transform(df_train_data.values)
            pt.fit(train_data_scaled)
            train_data_scaled = pt.transform(train_data_scaled)
        else:
            scaler.partial_fit(df_train_data.values)
            train_data_scaled = scaler.transform(df_train_data.values)
            train_data_scaled = pt.transform(train_data_scaled)

        train_data_scaled = np.clip(train_data_scaled / 3, -1, 1)
        train_data = torch.from_numpy(train_data_scaled)

        train_data = get_4D_shape(
            train_data,
            sequence_length=sequence_length,
            step=train_sequence_step,
        )
        train_labels = get_4D_shape(
            torch.from_numpy(df_train_labels.values),
            sequence_length=sequence_length,
            step=train_sequence_step,
        )
        train_timeline = df_train.index

        test_data_scaled = np.clip(
            pt.transform(scaler.transform(df_test_data.values)) / 3, -1, 1
        )
        test_data = torch.from_numpy(test_data_scaled)

        test_data = get_4D_shape(
            test_data, sequence_length=sequence_length, step=test_sequence_step
        )
        test_labels = get_4D_shape(
            torch.from_numpy(df_test_labels.values),
            sequence_length=sequence_length,
            step=test_sequence_step,
        )
        test_timeline = df_test.index

        train_concept_dataset = TemporalConceptDataset(
            TemporalConcept(
                name=f"train_{i}",
                data=train_data,
                labels=train_labels,
                timeline=train_timeline,
            )
        )

        test_concept_dataset = TemporalConceptDataset(
            TemporalConcept(
                name=f"test_{i}",
                data=test_data,
                labels=test_labels,
                timeline=test_timeline,
            )
        )

        # Attach the per-dataset stride so downstream evaluation code can map
        # flat (window, position) indices back to raw timesteps without having
        # to re-derive the mapping. This is needed for per-timestep score
        # aggregation when test windows overlap (test_sequence_step < seq_len).
        # test_sequence_step is also known as stride.
        train_concept_dataset.step = train_sequence_step
        test_concept_dataset.step = test_sequence_step

        train_concept_datasets.append(train_concept_dataset)
        test_concept_datasets.append(test_concept_dataset)

    return train_concept_datasets, test_concept_datasets


def get_domain_incremental_concept_datasets(
    domain_dfs: list[pd.DataFrame],
    value_column: str,
    label_column: str,
    sequence_length: int,
    train_sequence_step: int,
    test_sequence_step: int,
) -> tuple[list[TemporalConceptDataset], list[TemporalConceptDataset]]:
    train_concept_datasets: list[TemporalConceptDataset] = []
    test_concept_datasets: list[TemporalConceptDataset] = []

    for i, df in enumerate(domain_dfs):
        df = df.reset_index(drop=True)

        df_clean = get_df_without_anomalies(
            df=df.copy(), value_column=value_column, label_column=label_column,
        ).bfill()
        clean_data = get_df_data(df_clean, value_column).values
        orig_data = get_df_data(df, value_column).values
        clean_labels = get_df_labels(df_clean, label_column).values
        orig_labels = get_df_labels(df, label_column).values

        scaler = MinMaxScaler()
        pt = PowerTransformer(method="yeo-johnson")
        clean_scaled = scaler.fit_transform(clean_data)
        pt.fit(clean_scaled)
        clean_scaled = pt.transform(clean_scaled)
        clean_scaled = np.clip(clean_scaled / 3, -1, 1)

        orig_scaled = pt.transform(scaler.transform(orig_data))
        orig_scaled = np.clip(orig_scaled / 3, -1, 1)

        train_data = get_4D_shape(torch.from_numpy(clean_scaled),
                                   sequence_length=sequence_length,
                                   step=train_sequence_step)
        train_labels = get_4D_shape(torch.from_numpy(clean_labels),
                                     sequence_length=sequence_length,
                                     step=train_sequence_step)
        test_data = get_4D_shape(torch.from_numpy(orig_scaled),
                                  sequence_length=sequence_length,
                                  step=test_sequence_step)
        test_labels = get_4D_shape(torch.from_numpy(orig_labels),
                                    sequence_length=sequence_length,
                                    step=test_sequence_step)

        train_ds = TemporalConceptDataset(TemporalConcept(
            name=f"train_d{i}", data=train_data, labels=train_labels,
            timeline=df.index,
        ))
        test_ds = TemporalConceptDataset(TemporalConcept(
            name=f"test_d{i}", data=test_data, labels=test_labels,
            timeline=df.index,
        ))
        train_ds.step = train_sequence_step
        test_ds.step = test_sequence_step

        train_concept_datasets.append(train_ds)
        test_concept_datasets.append(test_ds)

    return train_concept_datasets, test_concept_datasets
