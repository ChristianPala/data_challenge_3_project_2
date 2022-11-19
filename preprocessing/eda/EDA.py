# Data Analysis:
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
import numpy as np
# Plotting:
import matplotlib.pyplot as plt
import seaborn as sns

# Global variables:
plot_path: Path = Path('plots')
data_path: Path = Path('..', '..', 'data')


# Functions:
def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot the correlation matrix.
    :param df: pd.DataFrame: the dataframe containing the dataset.
    :return: None
    """
    # select only the numerical features:
    numerical_features = df.select_dtypes(exclude='category').columns
    # exclude the id column:
    numerical_features = numerical_features.drop('id')
    # compute the correlation matrix:
    corr_matrix = df[numerical_features].corr('spearman')
    # mask the upper triangle:
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # plot the correlation matrix:
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True,
                cmap='coolwarm', fmt='.2f', annot_kws={'size': 10}, linewidths=0.5, mask=mask)
    # Annotate the correlation matrix:
    plt.title('Correlation Matrix with Spearman Correlation')

    # Save the plot:
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_path, 'correlation_matrix.png'))
    plt.show()


def plot_categorical_feature_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of the categorical features in the dataset.
    :param df: pd.DataFrame: the dataframe containing the dataset.
    :return: None
    """
    for feature in df.columns:
        if df[feature].dtype == 'category' and feature != 'default':
            plt.figure(figsize=(10, 10))
            sns.countplot(x=feature, data=df)
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plot_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(plot_path, f'{feature}_distribution.png'))
            plt.show()


def plot_numerical_feature_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of the numerical features in the dataset.
    @param df: pd.DataFrame: the dataframe containing the dataset.
    :return: None
    """
    for feature in df.columns:
        if df[feature].dtype != 'category' and feature != 'id':
            plt.figure(figsize=(10, 10))
            sns.histplot(x=feature, data=df, kde=True, bins='auto')
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plot_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(plot_path, f'{feature}_distribution.png'))
            plt.show()


def main() -> None:
    """
    Main function.
    :return: None. Plot the correlation matrix and the distribution of the categorical features.
    """
    # Load the pickle file:
    df: pd.DataFrame = pd.read_pickle(Path(data_path, 'project_2_dataset.pkl'))

    # Plot the correlation matrix:
    plot_correlation_matrix(df)

    # Plot the distribution of the categorical features:
    plot_categorical_feature_distribution(df)

    # Plot the distribution of the numerical features:
    plot_numerical_feature_distribution(df)

    # TODO: study the target variable and the features that are highly correlated with it.


if __name__ == '__main__':
    main()

