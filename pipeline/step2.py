import pprint as pp
import json
from modelbuilder.FeatureBuilder import FeatureBuilder
from modelbuilder.ModelBuilder import ModelBuilder
from datasets.RiverFlow.utils import (
    load_river_flow_data,
    create_river_flow_dataframe,
    river_flow_feature_extractors,
)


def get_best_model(file, model_type, autoregressive):
    with open(file) as f:
        data = json.load(f)

    best_model = None
    best_r2_adj = float('-inf')

    for model in data:
        if model['type'] == model_type and model['autoregressive'] == autoregressive:
            r2_adj = model['metrics']['test']['r2_adj']
            if r2_adj > best_r2_adj:
                best_model = model
                best_r2_adj = r2_adj

    return best_model


def main():
    """
    Loads the best model from json, initializes river flow dataframe, extracts features, builds the model,
    validates it, and saves its predictions.
    """

    model = get_best_model(
        file="models.json",
        model_type="RandomForestRegressor",
        autoregressive=False,
    )

    data = load_river_flow_data("../datasets/RiverFlow/data.pkl")
    df = create_river_flow_dataframe(data)

    feature_builder = FeatureBuilder(
        df=df,
        target="next_day_flow",
    )

    # Based on experiment, these are the necessary windows and lags to build the best models
    normal_windows = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 60, 90, 365]
    normal_lags = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20, 25, 30, 60, 90, 365]
    autoregressive_windows = [1, 2, 3, 4, 5, 7]
    autoregressive_lags = [1, 2]

    feature_builder.build_features(
        feature_extractors=river_flow_feature_extractors,
        windows=normal_windows,
        lags=normal_lags,
        autoregressive=model["autoregressive"]
    )

    model_builder = ModelBuilder(
        df=feature_builder.df,
        target=feature_builder.target,
        model_type=model["type"],
        autoregressive=model["autoregressive"],
        random_seed=2
    )

    features = model["features"]
    _, model_data = model_builder.build_model(features=features, test_size=0.3)
    model_builder.validate_model(n_splits=[3])
    pp.pprint(model_data)

    model_builder.save_predictions()


if __name__ == '__main__':
    main()
