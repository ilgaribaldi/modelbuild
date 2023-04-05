import os
import json
import time
import random
import numpy as np
from scipy import stats
from sklearn.svm import SVR
from joblib import Parallel, delayed
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pandas.errors import SettingWithCopyWarning, PerformanceWarning
import warnings
import pprint as pp


class FeatureBuilder:

    def __init__(self, df, target, verbose=False):
        self.df = df
        self.target = target
        self.feature_sets = []
        self.top_features = None
        self.verbose = verbose

    def build_features(
            self,
            feature_extractors,
            autoregressive=False,
            lags=None,
            windows=None,
    ):
        with warnings.catch_warnings():
            warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)
            warnings.simplefilter(action='ignore', category=PerformanceWarning)

            # Create a dictionary containing the variable names and their values
            params_mapping = {
                'df': self.df,
                'autoregressive': autoregressive,
                'lags': lags,
                'windows': windows,
                'verbose': self.verbose,
            }

            for feature_extractor in feature_extractors:
                func_params = feature_extractor.__code__.co_varnames[:feature_extractor.__code__.co_argcount]

                # Construct the kwargs dictionary by filtering the keys in func_params
                kwargs = {key: value for key, value in params_mapping.items() if key in func_params}

                feature_extractor(**kwargs)

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


class ModelBuilder:

    def __init__(self, df, target, model_type, autoregressive=False, random_seed=0):
        self.df = df
        self.model_type = model_type
        self.autoregressive = autoregressive
        self.target = target
        self.model = None
        self.y = df[self.target]
        self.X = None
        self.y_train = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.random_seed = random_seed
        random.seed(self.random_seed)

    # ------------------------ Initialization ----------------------- #
    def tuned_gradient_boosting_regressor(self):
        param_grid = {
            "n_estimators": [100, 200, 300],
            "learning_rate": [0.1, 0.05, 0.01],
            "max_depth": [3, 4, 5],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
            "subsample": [0.5, 0.8, 1.0]
        }

        grid_search = GridSearchCV(
            estimator=GradientBoostingRegressor(random_state=42),
            param_grid=param_grid,
            scoring="neg_mean_squared_error",
            cv=5,
            n_jobs=-1,
            verbose=2
        )

        grid_search.fit(self.X_train, self.y_train)

        # Get the best parameters
        best_params = grid_search.best_params_
        print("Best parameters found:", best_params)

        # Use the best parameters to initialize the model
        return GradientBoostingRegressor(**best_params, random_state=42)

    def initialize_model(self):
        print(self.model_type)
        model_map = {
            'GradientBoostingRegressor': GradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=5,
                max_features="log2",
                min_samples_leaf=4,
                n_estimators=100,
                subsample=0.5
            ),
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'LassoCV': LassoCV(cv=5, random_state=42, max_iter=10000),
            'SVR': SVR(kernel="poly"),
        }
        if self.model_type not in model_map:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model_map[self.model_type]

    def split_data(self, test_size):
        cutoff_index = int(len(self.X) * (1 - test_size))
        self.X_train = self.X.iloc[:cutoff_index]
        self.X_test = self.X.iloc[cutoff_index:]
        self.y_train = self.y.iloc[:cutoff_index]
        self.y_test = self.y.iloc[cutoff_index:]

    # ------------------------ Fit & Predict ------------------------ #

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred_test = self.model.predict(self.X_test).ravel()
        self.y_pred_train = self.model.predict(self.X_train).ravel()

    # --------------------------  Metrics ---------------------------- #

    def calculate_cross_val_metrics(self, n_splits, alpha=0.05):
        tscv = TimeSeriesSplit(n_splits=n_splits)

        def calculate_fold_metrics(train_index, test_index):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]
            self.model.fit(X_train, y_train)
            y_pred_test = self.model.predict(X_test).ravel()

            mse_test = mean_squared_error(y_test, y_pred_test)
            mae_test = mean_absolute_error(y_test, y_pred_test)
            r2_test = r2_score(y_test, y_pred_test)

            n_test = len(y_test)
            p_test = X_test.shape[1]
            r2_adj_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_test - 1)

            se_test = np.std(y_test - y_pred_test) / np.sqrt(n_test)
            ci_test = stats.t.interval(1 - alpha, df=n_test - 1, loc=0, scale=se_test)
            ci_avg_width = (ci_test[1] - ci_test[0]) / 2

            return {
                "mse": mse_test,
                "mae": mae_test,
                "r2": r2_test,
                "r2_adj": r2_adj_test,
                "ci_avg_width": ci_avg_width
            }

        fold_metrics = Parallel(n_jobs=n_splits)(
            delayed(calculate_fold_metrics)(train_index, test_index)
            for train_index, test_index in tscv.split(self.X)
        )

        return {
            "mse": np.mean([fm["mse"] for fm in fold_metrics]),
            "mae": np.mean([fm["mae"] for fm in fold_metrics]),
            "r2": np.mean([fm["r2"] for fm in fold_metrics]),
            "r2_adj": np.mean([fm["r2_adj"] for fm in fold_metrics]),
            "ci_avg_width": np.mean([fm["ci_avg_width"] for fm in fold_metrics])
        }

    def calculate_metrics(self, alpha=0.05):
        mse_train = mean_squared_error(self.y_train, self.y_pred_train)
        mae_train = mean_absolute_error(self.y_train, self.y_pred_train)
        r2_train = r2_score(self.y_train, self.y_pred_train)

        mse_test = mean_squared_error(self.y_test, self.y_pred_test)
        mae_test = mean_absolute_error(self.y_test, self.y_pred_test)
        r2_test = r2_score(self.y_test, self.y_pred_test)

        overfitting = mse_test - mse_train

        # Confidence interval calculations
        n_train = len(self.y_train)
        n_test = len(self.y_test)
        se_train = np.std(self.y_train - self.y_pred_train) / np.sqrt(n_train)
        se_test = np.std(self.y_test - self.y_pred_test) / np.sqrt(n_test)

        ci_train = stats.t.interval(1 - alpha, df=n_train - 1, loc=0, scale=se_train)
        ci_test = stats.t.interval(1 - alpha, df=n_test - 1, loc=0, scale=se_test)

        ci_avg_width_train = (ci_train[1] - ci_train[0]) / 2
        ci_avg_width_test = (ci_test[1] - ci_test[0]) / 2

        # Calculate the number of predictors
        p_train = self.X_train.shape[1]
        p_test = self.X_test.shape[1]

        # Calculate the adjusted R-squared values
        r2_adj_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p_train - 1)
        r2_adj_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_test - 1)

        # metrics_3_fold = self.calculate_cross_val_metrics(n_splits=3, alpha=alpha)
        # metrics_5_fold = self.calculate_cross_val_metrics(n_splits=5, alpha=alpha)
        # metrics_10_fold = self.calculate_cross_val_metrics(n_splits=10, alpha=alpha)

        return {
            "train": {
                "mse": mse_train,
                "mae": mae_train,
                "r2": r2_train,
                "r2_adj": r2_adj_train,
                "ci_avg_width": ci_avg_width_train,
            },
            "test": {
                "mse": mse_test,
                "mae": mae_test,
                "r2": r2_test,
                "r2_adj": r2_adj_test,
                "ci_avg_width": ci_avg_width_test,
            },
            "overfitting": overfitting,
        }

    # ---------------------------- Build ----------------------------- #

    @staticmethod
    def save_model(model_data, file='data/models2.json', verbose=False):
        # Load existing models from the file or initialize an empty list
        existing_models = []
        if os.path.exists(file):
            with open(file, 'r') as f:
                existing_models = json.load(f)

        # Sort the features alphabetically
        model_data['features'] = sorted(model_data['features'])

        # Check if a model with the same features and type exists
        for existing_model in existing_models:
            if (set(existing_model['features']) == set(model_data['features']) and
                    existing_model['type'] == model_data['type'] and
                    existing_model['random_seed'] == model_data['random_seed']
            ):
                if verbose:
                    print(f"{model_data['type']} model with {len(model_data['features'])} features SKIPPED.")
                return

        # Append the new model to the list and save to the file
        existing_models.append(model_data)
        with open(file, 'w') as f:
            json.dump(existing_models, f, indent=4)

        if verbose:
            print(f"{model_data['type']} model with {len(model_data['features'])} features SAVED.")

    def build_model(self, features, test_size, verbose=False):
        # Split data
        self.X = self.df[features]
        self.split_data(test_size=test_size)

        # Initialize model
        self.model = self.initialize_model()

        # Fit model
        if verbose:
            print(f"Fitting {self.model_type} model...")

        start_time = time.time()
        self.fit_model()
        time_to_build = time.time() - start_time

        # Predict
        self.predict()

        # Calculate Metrics
        metrics = self.calculate_metrics()

        feature_importances = (dict(zip(features, self.model.feature_importances_))
                               if hasattr(self.model, 'feature_importances_') else None)

        model_data = {
            "features": features,
            "feature_importances": feature_importances,
            "type": self.model_type,
            "autoregressive": self.autoregressive,
            "metrics": metrics,
            "time_to_build": time_to_build,
            "random_seed": self.random_seed,
        }

        return self.model, model_data

    def build_models(self, test_size, verbose=False, feature_sets=None):
        for index, features in enumerate(feature_sets, start=1):
            if verbose:
                print(f"Building model {index}...")

            # Call the build_model method
            _, model_data = self.build_model(features=features, test_size=test_size, verbose=verbose)

            self.save_model(model_data, verbose=verbose)

    # --------------------------------------------------------------- #
