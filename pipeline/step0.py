import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from datasets.RiverFlow.utils import load_river_flow_data, create_river_flow_dataframe


def plot_scatterplot(df, x, y):
    fig, ax = plt.subplots(figsize=(8, 5))

    # Select the x and y variables from the DataFrame
    x_data = df[x]
    y_data = df[y]

    # If x is '30d_average_temperature', calculate the rolling average
    if x == '30d_average_temperature':
        x_data = x_data.rolling(window=30).mean()
        df.dropna(inplace=True)

    sns.regplot(
        x=x_data.iloc[::10],
        y=y_data.iloc[::10],
        scatter=True,
        lowess=True,
        ax=ax,
        line_kws={'color': 'red'},
    )
    ax.set_xlabel('30-day Average Precipitation (mm)' if x == '30d_average_precipitation' else '30-Day Average Daily Temperature (°C)')
    ax.set_ylabel('River Flow (m$^{3}$/s)')
    ax.set_title(f'Relationship between {x} and River Flow')
    ax.grid()


def plot_timeseries(df, lower_bound, higher_bound, columns, normalize=False):
    # Convert the bounds to datetime objects
    lower_bound = pd.to_datetime(lower_bound)
    higher_bound = pd.to_datetime(higher_bound)

    # Filter the DataFrame based on the given bounds
    filtered_df = df[(df.index >= lower_bound) & (df.index <= higher_bound)]

    # Normalize the data if requested
    if normalize:
        scaler = MinMaxScaler()
        normalized_data = scaler.fit_transform(filtered_df[columns])
        normalized_df = pd.DataFrame(normalized_data, columns=columns, index=filtered_df.index)
    else:
        normalized_df = filtered_df

    # Set the Seaborn style and context
    sns.set_style("darkgrid")
    sns.set_context("notebook")

    # Create the plot
    plt.figure(figsize=(12, 3))

    # Plot each column independently
    colors = ['#0077BE',  '#67B6CF', 'orange', 'red',  '#67B6CF', '#FFA500']

    for i, column in enumerate(columns):
        sns.lineplot(x=normalized_df.index, y=normalized_df[column], label=column.capitalize(), color=colors[i], linewidth=2)

    # Set the x-axis label and y-axis label
    plt.xlabel("Time", fontsize=14)
    plt.ylabel("" if normalize else "Flow (m$^{3}$/s)", fontsize=14)

    # Add the legend and display the plot
    if len(columns) > 1:
        plt.legend(fontsize=12, loc='upper left')


def main():
    """
    loads river flow data, creates a dataframe, calculates some explanatory variables, then generates
    some useful plots for data visualization
    """

    # Initialize data
    data = load_river_flow_data("../datasets/RiverFlow/data.pkl")
    df = create_river_flow_dataframe(data)
    df['30d_average_temperature'] = df[[f"t_{i}" for i in range(9)]].rolling(window=30).mean().mean(axis=1)
    df['30d_average_precipitation'] = df[[f"p_{i}" for i in range(9)]].rolling(window=30).mean().mean(axis=1)
    df.dropna(inplace=True)

    # Set plotting style
    sns.set_style("dark")

    # Plot
    plot_timeseries(
        df=df,
        lower_bound='2005-01-01',
        higher_bound='2009-12-31',
        columns=['flow'],
        normalize=False
    )

    plot_scatterplot(
        df=df,
        x='30d_average_temperature',
        y='flow'
    )

    plot_scatterplot(
        df=df,
        x='30d_average_precipitation',
        y='flow'
    )

    plt.show()


if __name__ == '__main__':
    main()



