import random
import warnings
import numpy as np
import pickle
import pandas as pd
from pandas.errors import SettingWithCopyWarning, PerformanceWarning
from concurrent.futures import ThreadPoolExecutor
import time


def load_data(file_path):
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data


def create_dataframe(data):
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


class FeatureBuilder:

    def __init__(self, df, target, windows, lags, autoregressive=False, verbose=False):
        self.df = df
        self.target = target
        self.autoregressive = autoregressive
        self.feature_sets = []
        self.top_features = None
        self.windows = windows
        self.lags = lags
        self.verbose = verbose
        print(windows)

    # -------------------------- Target -------------------------- #

    def add_next_day_flow(self):
        self.df["next_day_flow"] = self.df["flow"].shift(-1)
        self.df = self.df.iloc[:-1]

    # -------------------------- Features ------------------------- #

    def add_total_daily_precipitation(self):
        if self.verbose:
            print("adding total daily precipitation...")
        precipitation_columns = [f"p_{i}" for i in range(9)]
        self.df["tdp"] = self.df[precipitation_columns].sum(axis=1)
        self.df = self.df.dropna()

    def add_statistical_features(self):
        if self.verbose:
            print("adding statistical features...")
        precipitation_columns = [f"p_{i}" for i in range(9)]
        temperature_columns = [f"t_{i}" for i in range(9)]
        # statistical features for precipitation
        self.df[f"1d_mean_precipitation"] = self.df[precipitation_columns].mean(axis=1)
        self.df[f"1d_min_precipitation"] = self.df[precipitation_columns].min(axis=1)
        self.df[f"1d_max_precipitation"] = self.df[precipitation_columns].max(axis=1)
        self.df[f"1d_std_precipitation"] = self.df[precipitation_columns].std(axis=1)

        # statistical features for temperature
        self.df[f"1d_mean_temperature"] = self.df[temperature_columns].mean(axis=1)
        self.df[f"1d_min_temperature"] = self.df[temperature_columns].min(axis=1)
        self.df[f"1d_max_temperature"] = self.df[temperature_columns].max(axis=1)
        self.df[f"1d_std_temperature"] = self.df[temperature_columns].std(axis=1)

    def add_aggregated_statistical_features(self):
        if self.verbose:
            print("adding aggregated statistical features...")
        precipitation_columns = [f"p_{i}" for i in range(9)]
        temperature_columns = [f"t_{i}" for i in range(9)]

        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)

            stats_map = {
                "mean": lambda x: x.mean(),
                "min": lambda x: x.min(),
                "max": lambda x: x.max(),
                "std": lambda x: x.std()
            }

            for window in self.windows:
                # Aggregated cumulative feature for precipitation
                self.df[f"{window}d_cumulative_precipitation"] = (
                    self.df[precipitation_columns].rolling(window=window).sum().sum(axis=1)
                )
                for stat, func in stats_map.items():
                    self.df[f"{window}d_{stat}_precipitation"] = (
                        self.df[precipitation_columns].rolling(window=window).apply(func, raw=True).sum(axis=1)
                    )
                    self.df[f"{window}d_{stat}_temperature"] = (
                        self.df[temperature_columns].rolling(window=window).apply(func, raw=True).mean(axis=1)
                    )
                    if self.autoregressive:
                        # Aggregated statistical features for flow
                        self.df[f"{window}d_{stat}_flow"] = (
                            self.df["flow"].rolling(window=window).apply(func, raw=True)
                        )

    def add_rate_of_change_features(self):
        if self.verbose:
            print("adding rate of change features...")
        for i in range(9):
            self.df[f"delta_p_{i}"] = self.df[f"p_{i}"].diff()
            self.df[f"delta_t_{i}"] = self.df[f"t_{i}"].diff()

        self.df["delta_tdp"] = self.df["tdp"].diff()

        for window in self.windows:
            self.df[f"{window}d_delta_mean_temperature"] = (
                self.df[f"{window}d_mean_temperature"].diff()
            )

    def add_interaction_features(self):
        if self.verbose:
            print("adding interaction features...")
        precipitation_columns = [f"p_{i}" for i in range(9)]
        temperature_columns = [f"t_{i}" for i in range(9)]

        for i, p_col in enumerate(precipitation_columns):
            for t_col in temperature_columns:
                self.df[f"interaction_{p_col}_{t_col}"] = self.df[p_col] * self.df[t_col]

        for window in self.windows:
            self.df[f"interaction_tdp_{window}d_mean_temperature"] = (
                    self.df["tdp"] * self.df[f"{window}d_mean_temperature"]
            )

    def add_lagged_features(self):
        if self.verbose:
            print("adding lagged features")
        aggregated_statistical_features = []

        for window in self.windows:
            print(window)
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
            if self.autoregressive:
                aggregated_statistical_features.append(f"{window}d_mean_flow")

        for lag in self.lags:
            for aggr_stat_feature in aggregated_statistical_features:
                self.df[f"lag({aggr_stat_feature},{lag})"] = self.df[aggr_stat_feature].shift(lag)

            if self.autoregressive:
                self.df[f"lag(flow,{lag})"] = self.df["flow"].shift(lag)

        self.df = self.df.dropna()

    def add_seasonality_features(self):
        if self.verbose:
            print("adding seasonality features...")
        self.df['day_of_year'] = self.df.index.dayofyear
        self.df['day_of_year_sin'] = np.sin(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['day_of_year_cos'] = np.cos(2 * np.pi * self.df['day_of_year'] / 365)
        self.df['month'] = self.df.index.month
        self.df['month_sin'] = np.sin(2 * np.pi * self.df['month'] / 12)
        self.df['month_cos'] = np.cos(2 * np.pi * self.df['month'] / 12)
        self.df = self.df.dropna()

    # --------------------------- Actions -------------------------- #

    def build_features(self):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
            warnings.simplefilter(action='ignore', category=PerformanceWarning)
            self.add_next_day_flow()
            self.add_total_daily_precipitation()
            self.add_statistical_features()
            start = time.time()
            self.add_aggregated_statistical_features()
            print(f"time4: {time.time() - start}")
            start = time.time()
            self.add_rate_of_change_features()
            print(f"time5: {time.time() - start}")
            start = time.time()
            self.add_interaction_features()
            print(f"time6: {time.time() - start}")
            start = time.time()
            self.add_lagged_features()
            print(f"time7: {time.time() - start}")
            start = time.time()
            self.add_seasonality_features()
            print(f"time8: {time.time() - start}")
            if not self.autoregressive:
                self.df = self.df.drop(columns=['flow'])
            self.df = self.df.dropna()

    def get_random_feature_sets(self, amount, max_features, min_features=1):
        self.feature_sets = []
        features = [col for col in self.df.columns if col != self.target]
        while len(self.feature_sets) < amount:
            feature_amount = random.randint(min_features, min(len(features), max_features))
            combination = random.sample(features, feature_amount)
            if combination not in self.feature_sets:
                self.feature_sets.append(combination)

    def get_top_features(self, amount):
        correlation_matrix = self.df.corr(method='pearson')
        correlations = correlation_matrix[self.target].drop(self.target)
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        self.top_features = [feature for feature, correlation in sorted_correlations[:amount]]
        self.df = self.df[self.top_features + [self.target]]

    def build_top_feature_sets(self, amount):
        self.feature_sets = []
        if self.top_features is None:
            self.get_top_features(amount)
        feature_sets = []
        for i in range(amount):
            self.feature_sets.append(self.top_features[:i + 1])

    # --------------------------------------------------------------- #
