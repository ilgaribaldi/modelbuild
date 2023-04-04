import json
import pprint as pp
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px


def load_models_from_json(file_path):
    with open(file_path, 'r') as f:
        models = json.load(f)
    return models


def extract_metrics(models):
    metrics = []
    for model in models:
        row = {
            'type': model['type'],
            'train_mse': model['metrics']['train']['mse'],
            'train_mae': model['metrics']['train']['mae'],
            'train_r2': (1-model['metrics']['train']['r2'])*100,
            'test_mse': model['metrics']['test']['mse'],
            'test_mae': model['metrics']['test']['mae'],
            'test_r2': (1-model['metrics']['test']['r2'])*100,
            'overfitting': model['metrics']['overfitting']
        }
        metrics.append(row)
    return metrics


def get_mse_values(models):
    mse_values = []
    for model in models:
        mse_values.append(model['metrics']['test']['mse'])
    return mse_values


def plot_mse_histogram(mse_values):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(mse_values, bins=30)
    ax.set_title('Histogram of Mean Squared Error (MSE) Distribution')
    ax.set_xlabel('MSE')
    ax.set_ylabel('Frequency')
    plt.show()


def plot_metrics_scatter_matrix(metrics):
    df = pd.DataFrame(metrics)
    sns.set(style="ticks")
    sns.pairplot(df, hue="type", kind="kde")
    plt.show()


def compare_metrics_by_model_type(metrics):
    df = pd.DataFrame(metrics)
    agg_metrics = df.groupby('type').agg(['mean', 'min', 'max'])
    return agg_metrics


def plot_metrics_boxplot(metrics):
    df = pd.DataFrame(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, metric in enumerate(['train_mse', 'train_mae', 'train_r2', 'test_mse', 'test_mae', 'test_r2']):
        row, col = divmod(i, 3)
        sns.boxplot(x='type', y=metric, data=df, ax=axes[row, col])
    plt.tight_layout()
    plt.show()


def plot_metrics_violinplot(metrics):
    df = pd.DataFrame(metrics)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for i, metric in enumerate(['train_mse', 'train_mae', 'train_r2', 'test_mse', 'test_mae', 'test_r2']):
        row, col = divmod(i, 3)
        sns.violinplot(x='type', y=metric, data=df, ax=axes[row, col])
    plt.tight_layout()
    plt.show()


file_path = 'data/models.json'
models = load_models_from_json(file_path)
mse_values = get_mse_values(models)
plot_mse_histogram(mse_values)
metrics = extract_metrics(models)
agg_metrics = compare_metrics_by_model_type(metrics)
# plot_metrics_scatter_matrix(metrics)

heatmap_df = agg_metrics.groupby(level=0, axis=1).min()
heatmap_df.index.name = 'Metric'

plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_df, annot=True, cmap='coolwarm')
plt.title('Average Evaluation Metrics Across All Model Types')
plt.show()

plot_metrics_boxplot(metrics)
plot_metrics_violinplot(metrics)


