from modelbuilder.ModelBuilder import ModelBuilder
from modelbuilder.FeatureBuilder import FeatureBuilder
from datasets.RiverFlow.utils import (
    load_river_flow_data,
    create_river_flow_dataframe,
    river_flow_feature_extractors,
    extract_next_day_flow,
    add_features,
)


def main():
    """
    Loads river flow data, creates a dataframe, extracts features, builds models, and exports them to a file.
    """

    # Initialize data
    data = load_river_flow_data("../datasets/RiverFlow/data.pkl")
    df = create_river_flow_dataframe(data)

    # Initialize FeatureBuilder instance
    feature_builder = FeatureBuilder(
        df=df,
        target="next_day_flow",
        verbose=True
    )

    # build features based on specified feature extractors, lags, and windows.
    feature_builder.build_features(
        feature_extractors=river_flow_feature_extractors,
        lags=[5, 10, 15, 30],
        windows=[1, 2, 3, 4, 5],
        autoregressive=False
    )

    feature_builder.get_top_features(amount=50)
    feature_builder.build_feature_sets(amount=20, random_size=False)

    # Initialize ModelBuilder Instance
    model_builder = ModelBuilder(
        df=feature_builder.df,
        target=feature_builder.target,
        model_type="GradientBoostingRegressor",
        autoregressive=False,
        random_seed=6
    )

    # Build and export models to json
    model_builder.export_models(
        file="models.json",
        test_size=0.3,
        feature_sets=feature_builder.feature_sets,
        verbose=False,
    )


if __name__ == '__main__':
    main()
