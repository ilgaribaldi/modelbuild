from utils.feature_builder import load_data, create_dataframe, FeatureBuilder
from utils.model_builder import ModelBuilder
import pprint as pp


def main():
    data = load_data("data/data.pkl")
    df = create_dataframe(data)
    df.to_excel("checking.xlsx")

    feature_builder = FeatureBuilder(
        df=df,
        target="next_day_flow",
        autoregressive=False,
        windows=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25, 30],
        lags=[1, 2, 3, 4, 5, 10, 11, 12, 15, 20, 25, 30],
        verbose=True
    )
    feature_builder.build_features()
    print(feature_builder.df)

    # feature_builder.get_top_features(amount=200)
    feature_builder.get_random_feature_sets(amount=50, max_features=100, min_features=20)
    # original
    features = [
        "t_0",
        "t_1",
        "t_2",
        "t_3",
        "t_4",
        "t_5",
        "t_6",
        "t_7",
        "t_8",
        "p_0",
        "p_1",
        "p_2",
        "p_3",
        "p_4",
        "p_5",
        "p_6",
        "p_7",
        "p_8",
    ]

    model_builder = ModelBuilder(
        df=feature_builder.df,
        target=feature_builder.target,
        model_type="GradientBoostingRegressor",
        autoregressive=feature_builder.autoregressive,
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
