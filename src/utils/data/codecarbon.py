import pandas as pd


def parse_codecarbon_stats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df[
        [
            "timestamp",
            "project_name",
            "duration",
            "emissions_rate",
            "cpu_power",
            "ram_power",
            "cpu_energy",
            "ram_energy",
        ]
    ]

    df.loc[:, "energy_consumed"] = df["cpu_energy"] + df["ram_energy"]

    df = df[
        [
            "timestamp",
            "project_name",
            "duration",
            "emissions_rate",
            "energy_consumed",
        ]
    ]

    df.columns = [
        "timestamp",
        "scenario",
        "duration [s]",
        "emissions_rate [kg/s]",
        "energy_consumed [kWh]",
    ]

    return df
