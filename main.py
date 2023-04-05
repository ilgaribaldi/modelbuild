from builders import FeatureBuilder, ModelBuilder
from examples.RiverFlow.utils import (
    load_river_flow_data,
    create_river_flow_dataframe,
    river_flow_feature_extractors,
)


def main():
    data = load_river_flow_data("examples/RiverFlow/data.pkl")
    df = create_river_flow_dataframe(data)

    feature_builder = FeatureBuilder(
        df=df,
        target="next_day_flow",
        verbose=True
    )
    feature_builder.build_features(
        feature_extractors=river_flow_feature_extractors,
        windows=[1, 2, 3, 4, 5, 10, 15, 20, 25, 30],
        lags=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        autoregressive=False
    )
    print(feature_builder.df)

    # feature_builder.get_top_features(amount=200)
    # feature_builder.get_random_feature_sets(amount=10, max_features=400, min_features=100)
    feature_builder.get_top_features(amount=500)
    feature_builder.build_top_feature_sets(amount=300)

    model_builder = ModelBuilder(
        df=feature_builder.df,
        target=feature_builder.target,
        model_type="GradientBoostingRegressor",
        autoregressive=False,
        random_seed=17
    )

    # model, model_data = model_builder.build_model(features=features, test_size=0.4)
    # pp.pprint(model_data)

    # gradient done (regressive and none regressive)
    model_builder.build_models(
        feature_sets=feature_builder.feature_sets,
        test_size=0.3,
        verbose=True,
    )


if __name__ == '__main__':
    main()
