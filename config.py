from pathlib import Path

SEED: int = 42

TRAIN_SIZE: float = 0.7
TEST_SIZE: float = 0.3

USE_EXTRACTED_DATA: bool = True

WEEKLY_SPLIT_SIZE: int = 24 * 7
BI_WEEKLY_SPLIT_SIZE: int = 2 * WEEKLY_SPLIT_SIZE

SEQUENCE_LENGTH: int = 12
TRAIN_SEQUENCE_STEP: int = SEQUENCE_LENGTH
TEST_SEQUENCE_STEP: int = 1

N_FEATURES: int = 1
BATCH_SIZE: int = 8
EPOCHS: int = 100
LEARNING_RATE: float = 4e-3

EARLY_STOPPING_PATIENCE: int = 5
EARLY_STOPPING_MIN_DELTA: int = 1

SMD_EARLY_STOPPING_MIN_DELTA: int = 150
SMD_EPOCHS: int = 60
SMD_BATCH_SIZE: int = 32

STRB_NOVELTY_THRESHOLD: float = 0.20
STRB_BUFFER_SIZE: int = 500

# Yahoo A1 Benchmark
YAHOO_A1_CONSOLIDATED_DATA_PATH: Path = (
    Path("data") / "consolidated" / "yahoo_a1.parquet"
)
YAHOO_A1_VALUE_COLUMN: str = "VALUE"
YAHOO_A1_LABEL_COLUMN: str = "IS_ANOMALY"
YAHOO_A1_NUM_TIME_SERIES: int = 67
YAHOO_A1_ARTIFACTS_PATH: Path = Path("artifacts") / "yahoo" / "A1Benchmark"

# Yahoo A2 Benchmark
YAHOO_A2_CONSOLIDATED_DATA_PATH: Path = (
    Path("data") / "consolidated" / "yahoo_a2.parquet"
)
YAHOO_A2_VALUE_COLUMN: str = "VALUE"
YAHOO_A2_LABEL_COLUMN: str = "IS_ANOMALY"
YAHOO_A2_NUM_TIME_SERIES: int = 100
YAHOO_A2_ARTIFACTS_PATH: Path = Path("artifacts") / "yahoo" / "A2Benchmark"

# Yahoo A3 Benchmark
YAHOO_A3_CONSOLIDATED_DATA_PATH: Path = (
    Path("data") / "consolidated" / "yahoo_a3.parquet"
)
YAHOO_A3_VALUE_COLUMN: str = "VALUE"
YAHOO_A3_LABEL_COLUMN: str = "IS_ANOMALY"
YAHOO_A3_NUM_TIME_SERIES: int = 100
YAHOO_A3_ARTIFACTS_PATH: Path = Path("artifacts") / "yahoo" / "A3Benchmark"

# Yahoo A4 Benchmark
YAHOO_A4_CONSOLIDATED_DATA_PATH: Path = (
    Path("data") / "consolidated" / "yahoo_a4.parquet"
)
YAHOO_A4_VALUE_COLUMN: str = "VALUE"
YAHOO_A4_LABEL_COLUMN: str = "IS_ANOMALY"
YAHOO_A4_NUM_TIME_SERIES: int = 100
YAHOO_A4_ARTIFACTS_PATH: Path = Path("artifacts") / "yahoo" / "A4Benchmark"
