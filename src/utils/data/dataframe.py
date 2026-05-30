from collections.abc import Generator
from pathlib import Path

import pandas as pd

from config import (
    YAHOO_A1_CONSOLIDATED_DATA_PATH,
    YAHOO_A1_LABEL_COLUMN,
    YAHOO_A1_VALUE_COLUMN,
    YAHOO_A2_CONSOLIDATED_DATA_PATH,
    YAHOO_A2_LABEL_COLUMN,
    YAHOO_A2_VALUE_COLUMN,
    YAHOO_A3_CONSOLIDATED_DATA_PATH,
    YAHOO_A3_LABEL_COLUMN,
    YAHOO_A3_VALUE_COLUMN,
    YAHOO_A4_CONSOLIDATED_DATA_PATH,
    YAHOO_A4_LABEL_COLUMN,
    YAHOO_A4_VALUE_COLUMN,
)


def read_df(path: Path, value_column: str, label_column: str) -> pd.DataFrame:
    df = pd.read_parquet(path)

    return df[
        [
            col
            for col in df.columns
            if (col.endswith(value_column) or col.endswith(label_column))
        ]
    ]


def read_yahoo_a1_df() -> pd.DataFrame:
    return read_df(
        YAHOO_A1_CONSOLIDATED_DATA_PATH,
        YAHOO_A1_VALUE_COLUMN,
        YAHOO_A1_LABEL_COLUMN,
    )


def read_yahoo_a2_df() -> pd.DataFrame:
    return read_df(
        YAHOO_A2_CONSOLIDATED_DATA_PATH,
        YAHOO_A2_VALUE_COLUMN,
        YAHOO_A2_LABEL_COLUMN,
    )


def read_yahoo_a3_df() -> pd.DataFrame:
    return read_df(
        YAHOO_A3_CONSOLIDATED_DATA_PATH,
        YAHOO_A3_VALUE_COLUMN,
        YAHOO_A3_LABEL_COLUMN,
    )


def read_yahoo_a4_df() -> pd.DataFrame:
    return read_df(
        YAHOO_A4_CONSOLIDATED_DATA_PATH,
        YAHOO_A4_VALUE_COLUMN,
        YAHOO_A4_LABEL_COLUMN,
    )


def split_df(
    df: pd.DataFrame, split_size: int
) -> Generator[pd.DataFrame, None, None]:
    n = len(df)
    for i in range(0, n - split_size + 1, split_size):
        yield df[i : i + split_size]


def get_df_data(df: pd.DataFrame, value_column: str) -> pd.DataFrame:
    return df[[col for col in df.columns if col.endswith(value_column)]]


def get_df_labels(df: pd.DataFrame, label_column: str) -> pd.DataFrame:
    return df[[col for col in df.columns if col.endswith(label_column)]]


def read_df_without_anomalies(
    path: Path, value_column: str, label_column: str
) -> pd.DataFrame:
    df = read_df(path, value_column, label_column)
    return get_df_without_anomalies(df, value_column, label_column)


def get_df_without_anomalies(
    df: pd.DataFrame, value_column: str, label_column: str
) -> pd.DataFrame:
    n_time_series = df.shape[1] // 2

    for ts_id in range(n_time_series):
        ts_id += 1

        mask_anomalies = df[f"TS_{ts_id}_{label_column}"] == 1
        df.loc[:, f"TS_{ts_id}_{value_column}"] = (
            df[f"TS_{ts_id}_{value_column}"].where(~mask_anomalies).ffill()
        )
        df.loc[:, f"TS_{ts_id}_{label_column}"] = df[
            f"TS_{ts_id}_{label_column}"
        ].replace(1, 0)

    return df
