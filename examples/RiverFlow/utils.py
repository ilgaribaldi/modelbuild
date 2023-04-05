import pandas as pd
import pickle
import numpy as np

# ----------------------------- data Initialization -------------------------- #


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


# ------------------------------ Normal features ----------------------------- #


def extract_total_daily_precipitation(df):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    df["total_daily_p"] = df[precipitation_columns].sum(axis=1)
    df.dropna(inplace=True)


def extract_statistics(df):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    temperature_columns = [f"t_{i}" for i in range(9)]

    df[f"mean_p"] = df[precipitation_columns].mean(axis=1)
    df[f"min_p"] = df[precipitation_columns].min(axis=1)
    df[f"max_p"] = df[precipitation_columns].max(axis=1)
    df[f"std_p"] = df[precipitation_columns].std(axis=1)

    df[f"mean_t"] = df[temperature_columns].mean(axis=1)
    df[f"min_t"] = df[temperature_columns].min(axis=1)
    df[f"max_t"] = df[temperature_columns].max(axis=1)
    df[f"std_t"] = df[temperature_columns].std(axis=1)

    df.dropna(inplace=True)


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
        df[f"{window}daily_cumulative_p"] = (
            df[precipitation_columns].rolling(window=window).sum().sum(axis=1)
        )
        for stat, func in stats_map.items():
            df[f"{window}d_{stat}_precipitation"] = (
                df[precipitation_columns].rolling(window=window).apply(func, raw=True).sum(axis=1)
            )
            df[f"{window}d_{stat}_temperature"] = (
                df[temperature_columns].rolling(window=window).apply(func, raw=True).mean(axis=1)
            )


def extract_rate_of_changes(df):
    for i in range(9):
        df[f"delta_p_{i}"] = df[f"p_{i}"].diff()
        df[f"delta_t_{i}"] = df[f"t_{i}"].diff()

    if "total_daily_p" in df.columns:
        df["delta_total_daily_p"] = df["total_daily_p"].diff()


def extract_lagged_features(df, lags):
    original_features = [f"t_{i}" for i in range(9)] + [f"p_{i}" for i in range(9)]

    for lag in lags:
        for original_feature in original_features:
            df[f"lag({original_feature},{lag})"] = df[original_feature].shift(lag)

    df.dropna(inplace=True)


def extract_lagged_statistical_features(df, lags):
    statistical_features = [
        "mean_p", "min_p", "max_p", "std_p",
        "mean_t", "min_t", "max_t", "std_t"
    ]

    for lag in lags:
        for stat_feature in statistical_features:
            df[f"lag({stat_feature},{lag})"] = df[stat_feature].shift(lag)

    df.dropna(inplace=True)


def extract_lagged_aggregated_statistical_features(df, windows, lags):
    aggregated_statistical_features = []

    for window in windows:
        aggregated_statistical_features.append(f"{window}daily_cumulative_p")
        for stat in ["mean", "min", "max", "std"]:
            aggregated_statistical_features.append(f"{window}d_{stat}_precipitation")
            aggregated_statistical_features.append(f"{window}d_{stat}_temperature")

    for lag in lags:
        for agg_stat_feature in aggregated_statistical_features:
            df[f"lag({agg_stat_feature},{lag})"] = df[agg_stat_feature].shift(lag)

    df.dropna(inplace=True)


def extract_interactions(df):
    precipitation_columns = [f"p_{i}" for i in range(9)]
    temperature_columns = [f"t_{i}" for i in range(9)]

    for p_col in precipitation_columns:
        for t_col in temperature_columns:
            df[f"interaction_{p_col}_{t_col}"] = df[p_col] * df[t_col]


def extract_seasonality_features(df):
    df['day_of_year'] = df.index.dayofyear
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month'] = df.index.month
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df.dropna(inplace=True)


# -------------------------- Autoregressive features -------------------------- #


def keep_flow(df, autoregressive):
    if not autoregressive:
        df.drop(columns=["flow"], inplace=True)


def extract_lagged_flow(df, autoregressive, lags):
    if autoregressive:
        for lag in lags:
            df[f"lag(flow,{lag})"] = df["flow"].shift(lag)
        df.dropna(inplace=True)


def extract_mean_flow(df, autoregressive, windows):
    if autoregressive:
        stats_map = {
            "mean": lambda x: x.mean(),
            "min": lambda x: x.min(),
            "max": lambda x: x.max(),
            "std": lambda x: x.std()
        }
        for stat, func in stats_map.items():
            for window in windows:
                df[f"{window}d_{stat}_flow"] = (
                    df["flow"].rolling(window=window).apply(func, raw=True)
                )
        df.dropna(inplace=True)


# ----------------------------------------------------------------------------- #


river_flow_feature_extractors = [
    extract_next_day_flow,
    extract_total_daily_precipitation,
    extract_statistics,
    extract_aggregated_statistics,
    extract_rate_of_changes,
    extract_lagged_features,
    extract_lagged_statistical_features,
    extract_lagged_aggregated_statistical_features,
    extract_interactions,
    extract_seasonality_features,
    keep_flow,
    extract_lagged_flow,
    extract_mean_flow,
]


