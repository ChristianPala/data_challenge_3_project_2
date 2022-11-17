# Data Analysis:
# Libraries:
# Data manipulation:
import pandas as pd
import numpy as np
from os import mkdir
from pathlib import Path
# Plotting:
import matplotlib.pyplot as plt
import seaborn as sns


# Functions:
def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot the correlation matrix.
    :param df: pd.DataFrame: the dataframe containing the dataset.
    :return: None
    """
    # Plot the correlation matrix of the numerical features:
    plt.figure(figsize=(10, 10))
    numerical_features = ["LIMIT_BAL", "AGE", "BILL_AMT1", "BILL_AMT2", "BILL_AMT3",
                          "BILL_AMT4", "BILL_AMT5", "BILL_AMT6", "PAY_AMT1", "PAY_AMT2",
                          "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6"]
    sns.heatmap(df[numerical_features].corr(), annot=True, cmap='coolwarm')

    # save the plot:
    plot_path: Path = Path('plots')
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_path, 'correlation_matrix.png'))

    # show the plot:
    plt.show()

def plot_sex_distribution(df: pd.DataFrame) -> None:
    """
    Plot
    :param df: pd.DataFrame: the dataframe containing the dataset.
    :return: None
    """
    # Plot the SEX distribution
    plt.figure(figsize=(10, 10))
    plt.bar(df.SEX.unique(), df.SEX.value_counts(), color=['blue', 'red'])
    plt.title("Gender distribution")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    # save the plot:
    plot_path: Path = Path('plots')
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_path, 'sex_distribution.png'))

    # show the plot:
    plt.show()





if __name__ == '__main__':
    # Load the data:
    df: pd.DataFrame = pd.read_csv(Path('..', '..', 'data', 'Project_2_Dataset.csv'))

    # plot the correlation matrix:
    plot_correlation_matrix(df)

    # plot the sex distribution:
    plot_sex_distribution(df)
