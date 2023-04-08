import random
import warnings
from pandas.errors import SettingWithCopyWarning, PerformanceWarning
import pprint as pp


class FeatureBuilder:
    """
    A class that facilitates the creation and selection of features for a machine learning model.

    Attributes:
        df (pandas.DataFrame): The input DataFrame.
        target (str): The name of the target variable.
        features (list): The list of selected features.
        feature_sets (list): The list of sets of selected features.
        verbose (bool): A flag indicating whether to print progress messages.

    Methods:
        __init__(self, df, target, verbose=False):
            Initializes a new FeatureBuilder instance.

        build_features(self, feature_extractors, autoregressive=False, lags=None, windows=None):
            Builds a set of features using the provided feature extractor functions.

        set_features(self, features):
            Sets the list of selected features.

        get_random_feature_sets(self, amount, max_features, min_features=1):
            Gets a list of random feature sets of specified size.

        get_top_features(self, amount):
            Selects the top `amount` features based on their correlation with the target variable.

        build_feature_sets(self, amount, random_size=False):
            Builds a list of feature sets using the previously selected features.
    """

    def __init__(self, df, target, verbose=False):
        """
        Initializes a new FeatureBuilder instance.

        Args:
            df (pandas.DataFrame): The input DataFrame.
            target (str): The name of the target variable.
            verbose (bool, optional): A flag indicating whether to print progress messages. Defaults to False.
        """

        self.df = df
        self.target = target
        self.features = None
        self.feature_sets = []
        self.verbose = verbose

    def build_features(self, feature_extractors, **kwargs):
        """
        Builds a set of features using the provided feature extractor functions.

        Args:
            feature_extractors (list): A list of feature extractor functions.
            autoregressive (bool, optional): A flag indicating whether to include autoregressive features.
            lags (list, optional): A list of lag values for autoregressive features.
            windows (list, optional): A list of window sizes for rolling features.
        """

        # Set default values for optional arguments
        autoregressive = kwargs.get('autoregressive', False)
        lags = kwargs.get('lags', None)
        windows = kwargs.get('windows', None)

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
                func_kwargs = {key: value for key, value in params_mapping.items() if key in func_params}

                feature_extractor(**func_kwargs)

    def normalize_features(self):
        """
        Normalizes the selected features using Min-Max scaling.
        """

        if self.features is None:
            self.features = [col for col in self.df.columns if col != self.target]

        # Normalize the features
        for feature in self.features:
            min_value = self.df[feature].min()
            max_value = self.df[feature].max()
            self.df[feature] = (self.df[feature] - min_value) / (max_value - min_value)

        if self.verbose:
            print("Features have been normalized using Min-Max scaling.")

    def normalize_target(self):
        """
        Normalizes the target variable using Min-Max scaling.
        """
        if self.target is None:
            raise ValueError("No target variable is set. Please set the target variable before normalizing.")

        # Normalize the target variable
        min_value = self.df[self.target].min()
        max_value = self.df[self.target].max()
        self.df[self.target] = (self.df[self.target] - min_value) / (max_value - min_value)

        if self.verbose:
            print("Target variable has been normalized using Min-Max scaling.")

    def set_features(self, features):
        """
        Sets the list of selected features.

        Args:
            features (list): A list of feature names.
        """

        self.features = features
        self.df = self.df[self.features + [self.target]]

    def get_random_feature_sets(self, amount, **kwargs):
        """
        Gets a list of random feature sets of specified size.

        Args:
            amount (int): The number of feature sets to generate.
            max_features (int, optional): The maximum number of features in a feature set.
            min_features (int, optional): The minimum number of features in a feature set. Defaults to 1.
        """

        # Get features
        features = [col for col in self.df.columns if col != self.target]

        # Set default values for optional arguments
        min_features = kwargs.get('min_features', 1)
        max_features = kwargs.get('max_features', len(features))

        # Generate the feature sets
        self.feature_sets = []
        features = [col for col in self.df.columns if col != self.target]
        while len(self.feature_sets) < amount:
            feature_amount = random.randint(min_features, min(len(features), max_features))
            combination = random.sample(features, feature_amount)
            if combination not in self.feature_sets:
                self.feature_sets.append(combination)

    def get_top_features(self, amount):
        """
        Selects the top `amount` features based on their correlation with the target variable.

        Args:
            amount (int): The number of top features to select.
        """

        if self.verbose:
            print("calculating top features")

        correlation_matrix = self.df.corr(method='pearson')
        correlations = correlation_matrix[self.target].drop(self.target)
        sorted_correlations = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
        self.features = [feature for feature, correlation in sorted_correlations[:amount]]
        self.df = self.df[self.features + [self.target]]

    def build_feature_sets(self, amount, **kwargs):
        """
        Builds a list of feature sets using the previously selected features.

        Args:
            amount (int): The number of feature sets to generate.
            random_size (bool, optional): A flag indicating whether the feature sets should have random sizes. Defaults to False.
        """

        # Set default value for optional argument
        random_size = kwargs.get('random_size', False)

        self.feature_sets = []
        if self.features is None:
            self.get_top_features(amount)

        if random_size:
            for _ in range(amount):
                random_subset_size = random.randint(1, len(self.features))
                random_subset = random.sample(self.features, random_subset_size)
                self.feature_sets.append(random_subset)
        else:
            for i in range(amount):
                self.feature_sets.append(self.features[:i + 1])
