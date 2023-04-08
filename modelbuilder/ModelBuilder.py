import os
import json
import time
import random
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.svm import SVR
from joblib import Parallel, delayed
from sklearn.preprocessing import MinMaxScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LassoCV
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV, train_test_split


class ModelBuilder:
    """
    A class to simplify the process of building, fitting, predicting, and evaluating machine learning models.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.
        target (str): The name of the target variable.
        model_type (str): The type of model to build.
        autoregressive (bool): A flag indicating whether the model is autoregressive.
        random_seed (int): The random seed for reproducibility.
        model: The initialized model.
        model_data: A dictionary containing model information and performance metrics.
        y (pandas.Series): The target variable series.
        X (pandas.DataFrame): The feature matrix.
        y_train (pandas.Series): The target variable series for the training set.
        y_test (pandas.Series): The target variable series for the test set.
        X_train (pandas.DataFrame): The feature matrix for the training set.
        X_test (pandas.DataFrame): The feature matrix for the test set.
        y_pred_train (numpy.ndarray): The predicted target values for the training set.
        y_pred_test (numpy.ndarray): The predicted target values for the test set.

    Methods:
        __init__(self, df, target, model_type, autoregressive=False, random_seed=0):
            Initializes a new ModelBuilder instance.

        tuned_gradient_boosting_regressor(self):
            Performs a grid search for the best hyperparameters of a GradientBoostingRegressor model.

        initialize_model(self):
            Initializes the selected model based on the model_type attribute.

        split_data(self, test_size, respect_temporal_order=True):
            Splits the data into training and test sets based on the provided test_size.

        fit_model(self):
            Fits the initialized model on the training data.

        predict(self):
            Predicts target values for the training and test sets using the fitted model.

        calculate_fold_metrics(self, train_index, test_index, alpha=0.05):
            Calculates performance metrics for a specific fold during cross-validation.

        calculate_metrics(self, alpha=0.05):
            Calculates performance metrics for the fitted model on the training and test sets.

        build_model(self, features, test_size, verbose=False):
            Builds, fits, and evaluates the model using the provided features and test_size.

        export_models(self, output_file, test_size, verbose=False, feature_sets=None):
            Builds and saves multiple models with different feature sets to a file.

        validate_model(self, n_splits):
            Performs time series cross-validation and calculates mean performance metrics.

        save_model(model_data, file):
            Saves the model_data dictionary to a file.
    """

    def __init__(self, df, target, model_type, **kwargs):
        """
        Initializes a new ModelBuilder instance.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            target (str): The name of the target variable.
            model_type (str): The type of model to build.
            autoregressive (bool, optional): A flag indicating whether the model is autoregressive. Defaults to False.
            random_seed (int, optional): The random seed for reproducibility. Defaults to 0.
        """

        # Set default values for optional arguments
        autoregressive = kwargs.get('autoregressive', False)
        random_seed = kwargs.get('random_seed', 0)

        self.df = df
        self.model_type = model_type
        self.autoregressive = autoregressive
        self.target = target
        self.model = None
        self.model_data = None
        self.y = df[self.target]
        self.X = None
        self.y_train = None
        self.y_test = None
        self.X_train = None
        self.X_test = None
        self.y_pred_train = None
        self.y_pred_test = None
        self.random_seed = random_seed

        self.upper_bound = None
        self.lower_bound = None
        random.seed(self.random_seed)

    # --------------------------- Initialization -------------------------- #

    def tuned_gradient_boosting_regressor(self):
        """
        Performs a grid search for the best hyperparameters of a GradientBoostingRegressor model.

        Returns:
            GradientBoostingRegressor: The GradientBoostingRegressor model with the best hyperparameters.
        """

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
        """
        Initializes the selected model based on the model_type attribute.

        Returns:
            A model instance corresponding to the specified model_type.
        """

        if self.model_type == "TunedGradientBoostingRegressor":
            return self.tuned_gradient_boosting_regressor()

        model_map = {
            'DecisionTreeRegressor': DecisionTreeRegressor(random_state=42),
            'RandomForestRegressor': RandomForestRegressor(random_state=42),
            'LinearRegression': LinearRegression(),
            'Ridge': Ridge(random_state=42),
            'Lasso': Lasso(random_state=42),
            'LassoCV': LassoCV(cv=5, random_state=42, max_iter=10000),
            'SVR': SVR(kernel="poly"),
            'GradientBoostingRegressor': GradientBoostingRegressor(
                learning_rate=0.05,
                max_depth=5,
                max_features="sqrt",
                min_samples_leaf=4,
                min_samples_split=10,
                n_estimators=200,
                subsample=0.8
            ),
        }
        if self.model_type not in model_map:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        return model_map[self.model_type]

    def split_data(self, test_size, **kwargs):
        """
        Splits the data into training and testing sets.

        Args:
            test_size (float): The proportion of the dataset to include in the test split.
            respect_temporal_order (bool, optional): A flag indicating whether to respect the temporal order of the
                data when splitting. Defaults to True.
        """

        # Set default values for optional arguments
        respect_timeseries = kwargs.get('respect_timeseries', True)

        if respect_timeseries:
            cutoff_index = int(len(self.X) * (1 - test_size))
            self.X_train = self.X.iloc[:cutoff_index]
            self.X_test = self.X.iloc[cutoff_index:]
            self.y_train = self.y.iloc[:cutoff_index]
            self.y_test = self.y.iloc[cutoff_index:]
        else:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
                self.X,
                self.y,
                test_size=test_size,
                random_state=42
            )

    # --------------------------- Fit & Predict --------------------------- #

    def fit_model(self):
        """
        Fits the model to the training data.
        """

        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """
        Generates predictions for the training and testing data using the fitted model.
        """

        self.y_pred_test = self.model.predict(self.X_test).ravel()
        self.y_pred_train = self.model.predict(self.X_train).ravel()

    # ------------------------------ Metrics ------------------------------ #

    def calculate_fold_metrics(self, train_index, test_index, **kwargs):
        """
        Calculates performance metrics for a specific fold in cross-validation.

        Args:
            train_index (list): Indices of the training data.
            test_index (list): Indices of the testing data.
            alpha (float, optional): The significance level for confidence interval calculations. Defaults to 0.05.

        Returns:
            dict: A dictionary containing performance metrics for the fold.
        """

        # Set default values for optional arguments
        alpha = kwargs.get('alpha', 0.05)

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

    def calculate_metrics(self, **kwargs):
        """
        Calculates performance metrics for the training and testing using the fitted model.

        Args:
            model: The fitted model.
            alpha (float, optional): The significance level for confidence interval calculations. Defaults to 0.05.

        Returns:
            dict: A dictionary containing performance metrics for the training and testing data.
        """

        # Set default values for optional arguments
        alpha = kwargs.get('alpha', 0.05)

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

        residuals_test = self.y_test - self.y_pred_test
        residuals_var_test = np.var(residuals_test)
        residuals_std_test = np.sqrt(residuals_var_test)
        t_score_test = stats.t.ppf(1 - alpha / 2, df=n_test - 1)

        self.lower_bound = self.y_pred_test - t_score_test * residuals_std_test
        self.upper_bound = self.y_pred_test + t_score_test * residuals_std_test

        # Calculate the number of predictors
        p_train = self.X_train.shape[1]
        p_test = self.X_test.shape[1]

        # Calculate the adjusted R-squared values
        r2_adj_train = 1 - (1 - r2_train) * (n_train - 1) / (n_train - p_train - 1)
        r2_adj_test = 1 - (1 - r2_test) * (n_test - 1) / (n_test - p_test - 1)

        return {
            "train": {
                "mse": mse_train,
                "mae": mae_train,
                "r2": r2_train,
                "r2_adj": r2_adj_train,
            },
            "test": {
                "mse": mse_test,
                "mae": mae_test,
                "r2": r2_test,
                "r2_adj": r2_adj_test,
            },
            "overfitting": overfitting,
        }

    # ------------------------------- Build & Save -------------------------------- #

    def save_model_data(self, file):
        """
        Saves the trained model data to the specified file.

        Args:
            file (str): The name of the file where the model data will be saved.

        This method checks if a model with the same features and type already exists in the file.
        If a matching model is found, it will be skipped. Otherwise, the new model data will be appended
        to the list of existing models and saved to the file.
        """

        # Load existing models from the file or initialize an empty list
        existing_models = []
        if os.path.exists(file):
            with open(file, 'r') as f:
                existing_models = json.load(f)

        # Sort the features alphabetically
        self.model_data['features'] = sorted(self.model_data['features'])

        # Check if a model with the same features and type exists
        for existing_model in existing_models:
            if (set(existing_model['features']) == set(self.model_data['features']) and
                    existing_model['type'] == self.model_data['type'] and
                    existing_model['random_seed'] == self.model_data['random_seed']):
                print(f"{self.model_data['type']} model with {len(self.model_data['features'])} features SKIPPED.")
                return

        # Append the new model to the list and save to the file
        existing_models.append(self.model_data)
        with open(file, 'w') as f:
            json.dump(existing_models, f, indent=4)
            print(f"{self.model_data['type']} model with {len(self.model_data['features'])} features SAVED.")

    def save_predictions(self):
        y_pred_test = pd.Series(self.y_pred_test, index=self.y_test.index, name="predicted")
        lower_bound = pd.Series(self.lower_bound, index=self.y_test.index, name="lower_bound")
        upper_bound = pd.Series(self.upper_bound, index=self.y_test.index, name="upper_bound")
        prediction_df = pd.concat([self.y_test, y_pred_test, lower_bound, upper_bound], axis=1)
        prediction_df.to_parquet('predictions.parquet', index=True)

    def build_model(self, features, test_size, **kwargs):
        """
        Builds, fits, and evaluates a machine learning model using the specified features and test size.

        Args:
            features (list): The list of selected features.
            test_size (float): The proportion of the dataset to include in the test split.
            verbose (bool, optional): A flag indicating whether to print progress messages. Defaults to False.
            normalize (bool, optional): A flag indicating whether to normalize the data. Defaults to False.

        Returns:
            tuple: A tuple containing the built model and a dictionary with model data and performance metrics.
        """

        # Set default values for optional arguments
        verbose = kwargs.get('verbose', False)
        normalize = kwargs.get('normalize', True)
        alpha = kwargs.get('alpha', 0.05)

        # Split data
        self.X = self.df[features]
        self.split_data(test_size=test_size)

        # Normalize data if normalize is set to True
        if normalize:
            scaler = MinMaxScaler()
            self.X_train = scaler.fit_transform(self.X_train)
            self.X_test = scaler.transform(self.X_test)

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
        metrics = self.calculate_metrics(alpha=alpha)

        feature_importances = (dict(zip(features, self.model.feature_importances_))
                               if hasattr(self.model, 'feature_importances_') else None)

        self.model_data = {
            "features": features,
            "type": self.model_type,
            "autoregressive": self.autoregressive,
            "metrics": metrics,
            "time_to_build": time_to_build,
            "random_seed": self.random_seed,
        }

        return self.model, self.model_data

    def export_models(self, file, feature_sets, **kwargs):
        """
        Builds, fits, and evaluates multiple machine learning models using different feature sets and saves the results
        to a file.

        Args:
            file (str): The path to the output file.
            feature_sets (list): The list of sets of selected features.
            test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.3.
            verbose (bool, optional): A flag indicating whether to print progress messages. Defaults to False.
        """

        # Set default values for optional arguments
        test_size = kwargs.get('test_size', 0.3)
        verbose = kwargs.get('verbose', False)

        for index, features in enumerate(feature_sets, start=1):
            if verbose:
                print(f"Building model {index}...")

            # Call the build_model method
            _, self.model_data = self.build_model(features=features, test_size=test_size, verbose=verbose)

            self.save_model_data(file)

    # ------------------------------- Validate ----------------------------- #

    def validate_model(self, n_splits):
        """
        Validates the model using cross-validation with the specified number of splits.

        Args:
            n_splits (list): A list of integers specifying the number of splits to use for cross-validation.

        Returns:
            dict: A dictionary containing the mean performance metrics for the n_splits in the list.
        """

        mean_fold_metrics = None

        for splits in n_splits:
            tscv = TimeSeriesSplit(n_splits=splits)
            fold_metrics = Parallel(n_jobs=splits)(
                delayed(self.calculate_fold_metrics)(train_index, test_index)
                for train_index, test_index in tscv.split(self.X)
            )

            # Calculate the mean fold metrics
            mean_fold_metrics = {
                "mse": np.mean([fm["mse"] for fm in fold_metrics]),
                "mae": np.mean([fm["mae"] for fm in fold_metrics]),
                "r2": np.mean([fm["r2"] for fm in fold_metrics]),
                "r2_adj": np.mean([fm["r2_adj"] for fm in fold_metrics]),
                "ci_avg_width": np.mean([fm["ci_avg_width"] for fm in fold_metrics])
            }

            # Insert the fold metrics into the metrics object
            self.model_data["metrics"][f"{splits}-fold"] = mean_fold_metrics

        # Return the mean fold metrics for the last n_splits in the list
        return mean_fold_metrics

    # ---------------------------------------------------------------------- #
