import pandas as pd
import pickle
import numpy as np

# ---------------------------- data Initialization --------------------------- #


def load_river_flow_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_river_flow_dataframe(data):
    flow_df = pd.DataFrame(
        data["flow"], columns=["flow"],
        index=pd.to_datetime(data["flow_dates"])
    )

    temperature_df = pd.DataFrame(
        data["obs_tas"], columns=[f"t_{i}" for i in range(9)],
        index=pd.to_datetime(data["obs_dates"])
    )

    precipitation_df = pd.DataFrame(
        data["obs_pr"], columns=[f"p_{i}" for i in range(9)],
        index=pd.to_datetime(data["obs_dates"])
    )

    df = pd.concat([flow_df, temperature_df, precipitation_df], axis=1)
    df.dropna(inplace=True)
    return df


# ---------------------------------- Target ---------------------------------- #


def extract_next_day_flow(df):
    df["next_day_flow"] = df["flow"].shift(-1)
    df.dropna(inplace=True)
    return df


# ------------------------------ Normal Features ----------------------------- #


def extract_total_daily_precipitation(df):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    df["tdp"] = df[precipitation_columns].sum(axis=1)
    return df


def extract_statistics(df):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    temperature_columns = [f"t_{i}" for i in range(9)]

    # statistical features for precipitation
    df[f"1d_mean_precipitation"] = df[precipitation_columns].mean(axis=1)
    df[f"1d_min_precipitation"] = df[precipitation_columns].min(axis=1)
    df[f"1d_max_precipitation"] = df[precipitation_columns].max(axis=1)
    df[f"1d_std_precipitation"] = df[precipitation_columns].std(axis=1)

    # statistical features for temperature
    df[f"1d_mean_temperature"] = df[temperature_columns].mean(axis=1)
    df[f"1d_min_temperature"] = df[temperature_columns].min(axis=1)
    df[f"1d_max_temperature"] = df[temperature_columns].max(axis=1)
    df[f"1d_std_temperature"] = df[temperature_columns].std(axis=1)

    df.dropna(inplace=True)
    return df


def extract_aggregated_statistics(df, windows):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    temperature_columns = [f"t_{i}" for i in range(9)]

    stats_map = {
        "mean": lambda x: x.mean(),
        "min": lambda x: x.min(),
        "max": lambda x: x.max(),
        "std": lambda x: x.std()
    }

    for window in windows:
        # Aggregated cumulative feature for precipitation
        df[f"{window}d_cumulative_precipitation"] = (
            df[precipitation_columns].rolling(window=window).sum().sum(axis=1)
        )
        for stat, func in stats_map.items():
            df[f"{window}d_{stat}_precipitation"] = (
                df[precipitation_columns].rolling(window=window).apply(func, raw=True).sum(axis=1)
            )
            df[f"{window}d_{stat}_temperature"] = (
                df[temperature_columns].rolling(window=window).apply(func, raw=True).mean(axis=1)
            )
    df.dropna(inplace=True)
    return df


def extract_rates_of_change(df, windows):
    for i in range(9):
        df[f"delta_p_{i}"] = df[f"p_{i}"].diff()
        df[f"delta_t_{i}"] = df[f"t_{i}"].diff()

    df["delta_tdp"] = df["tdp"].diff()

    for window in windows:
        df[f"{window}d_delta_mean_temperature"] = (
            df[f"{window}d_mean_temperature"].diff()
        )
    df.dropna(inplace=True)
    return df


def extract_interactions(df, windows):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    temperature_columns = [f"t_{i}" for i in range(9)]

    for i, p_col in enumerate(precipitation_columns):
        for t_col in temperature_columns:
            df[f"interaction_{p_col}_{t_col}"] = df[p_col] * df[t_col]

    for window in windows:
        df[f"interaction_tdp_{window}d_mean_temperature"] = (
                df["tdp"] * df[f"{window}d_mean_temperature"]
        )
    df.dropna(inplace=True)
    return df


def extract_lagged_features(df, windows, lags):
    aggregated_statistical_features = []

    for window in windows:
        aggregated_statistical_features.extend([
            f"{window}d_cumulative_precipitation",
            f"{window}d_mean_precipitation",
            f"{window}d_min_precipitation",
            f"{window}d_max_precipitation",
            f"{window}d_std_precipitation",
            f"{window}d_mean_temperature",
            f"{window}d_min_temperature",
            f"{window}d_max_temperature",
            f"{window}d_std_temperature",
        ])

    for lag in lags:
        for aggr_stat_feature in aggregated_statistical_features:
            df[f"lag({aggr_stat_feature},{lag})"] = df[aggr_stat_feature].shift(lag)
    df.dropna(inplace=True)
    return df


def extract_seasonality_features(df):
    df['day_of_year'] = df.index.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df.dropna(inplace=True)
    return df


# -------------------------- Autoregressive Features ------------------------- #


def keep_flow(df, autoregressive):
    if not autoregressive:
        df.drop(columns=["flow"], inplace=True)
    return df


def extract_autoregressive_aggregated_statistics(df, autoregressive, windows):
    if autoregressive:

        stats_map = {
            "mean": lambda x: x.mean(),
            "min": lambda x: x.min(),
            "max": lambda x: x.max(),
            "std": lambda x: x.std()
        }

        for window in windows:
            for stat, func in stats_map.items():
                # Aggregated statistical features for flow
                df[f"{window}d_{stat}_flow"] = (
                    df["flow"].rolling(window=window).apply(func, raw=True)
                )
    return df


def extract_lagged_autoregressive_features(df, autoregressive, windows, lags):
    if autoregressive:
        aggregated_autoregressive_features = []

        for window in windows:
            aggregated_autoregressive_features.append(f"{window}d_mean_flow")

        for lag in lags:
            for aggr_ar_feature in aggregated_autoregressive_features:
                df[f"lag({aggr_ar_feature},{lag})"] = df[aggr_ar_feature].shift(lag)

            df[f"lag(flow,{lag})"] = df["flow"].shift(lag)

    df.dropna(inplace=True)
    return df


# ---------------------------------------------------------------------------- #


river_flow_feature_extractors = [
    extract_next_day_flow,
    extract_total_daily_precipitation,
    extract_aggregated_statistics,
    extract_rates_of_change,
    extract_interactions,
    extract_lagged_features,
    extract_seasonality_features,
    keep_flow,
    extract_autoregressive_aggregated_statistics,
    extract_lagged_autoregressive_features,
]


