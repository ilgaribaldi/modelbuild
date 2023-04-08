import pandas as pd
import matplotlib.pyplot as plt
from pipeline.step0 import plot_timeseries
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def adjusted_r2(y_true, y_pred, n, k):
    r2 = r2_score(y_true, y_pred)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return adj_r2


def calculate_metrics(group):
    y_true = group['observed']
    y_pred = group['predicted']
    n = len(y_true)
    k = 1

    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    adj_r2 = adjusted_r2(y_true, y_pred, n, k)

    return pd.Series({'MSE': mse, 'MAE': mae, 'R2': r2, 'R2_adj': adj_r2})


def get_metrics_by_year(df):
    # Add a column with the year
    df['year'] = df.index.year
    df = df[df['year'] >= 2000]
    metrics_by_year = df.groupby('year').apply(calculate_metrics)

    # calculate average metrics
    average_metrics = metrics_by_year.mean()
    average_metrics.name = "Average"

    # Append the average metrics row to the DataFrame
    metrics_by_year.loc["Average"] = average_metrics
    rounded_metrics_by_year = metrics_by_year.round(2)
    return rounded_metrics_by_year


def main():
    """
    loads a parquet file containing predictions, observed values, upper & lower bound.
    It then calculates metrics by year using the 'get_metrics_by_year' function and prints the metrics.
    Finally, it plots a time series graph for any of the specified columns in a given time range,
    """

    # Load predictions, observed values, upper & lower bount
    df = pd.read_parquet('data/predictions.parquet')
    df = df.rename(columns={'next_day_flow': 'observed'})

    # calculate metrics by year
    metrics_by_year = get_metrics_by_year(df)
    print(metrics_by_year)

    # plot comparisons
    plot_timeseries(
            df=df,
            lower_bound='2000-01-01',
            higher_bound='2010-12-31',
            columns=['observed', 'predicted'],  # ['observed', 'predicted', 'lower_bound', 'upper_bound']
            normalize=False
        )

    plt.show()


if __name__ == '__main__':
    main()
