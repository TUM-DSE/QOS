
import pandas as pd
import numpy as np

def get_average(dataframes: list[pd.DataFrame], key: str, base_key: str | None = None) -> float:
    if base_key is not None:
        for df in dataframes:
            df["for_mean"] = df[key] / df[base_key]

    # concatenate all dataframes into one
    df = pd.concat(dataframes, ignore_index=True)
    return np.average(df["for_mean"])