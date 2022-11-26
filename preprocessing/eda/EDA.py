# Data Analysis:
# Libraries:
# Data manipulation:
from pathlib import Path
# Plotting:
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# Timing:
from auxiliary.method_timer import measure_time
# Global variables:
from config import plot_path, csv_file_path
if not plot_path:
    plot_path.mkdir(parents=True, exist_ok=True)


# Functions:
def category_mapping(feature_name: str) -> dict[int, str]:
    """
    Map the category labels to their corresponding values, auxiliary function for the legend
    in the categorical features distribution plots.
    @param feature_name: str: the name of the feature to be mapped.
    :return: Dict[int, str]: the mapping for the category labels.
    """
    if feature_name == 'gender':
        category_dict = {1: "male", 2: "female"}
    elif feature_name == 'education':
        category_dict = {1: "graduate school", 2: "university", 3: "high school", 4: "others"}
    elif feature_name == 'marriage':
        category_dict = {1: "married", 2: "single", 3: "others"}
    elif feature_name.startswith('pay_'):
        category_dict = {-2: "no consumption", -1: "paid fully", 0: "revolving credit", 1: "delay for 1 month",
                         2: "delay for 2 months", 3: "delay for 3 months", 4: "delay for 4 months",
                         5: "delay for 5 months", 6: "delay for 6 months", 7: "delay for 7 months",
                         8: "delay for 8 months", 9: "delay for 9 months or more"}

    elif feature_name == 'default':
        category_dict = {0: "no default", 1: "default"}
    else:
        raise ValueError(f'Invalid feature name: {feature_name}')

    return category_dict


def plot_correlation_matrix(df: pd.DataFrame) -> None:
    """
    Plot the correlation matrix.
    :param df: pd.DataFrame: the dataframe to compute the correlation matrix from.
    :return: None
    """
    # select only the numerical features:
    numerical_features = df.select_dtypes(exclude='category').columns
    # compute the correlation matrix:
    corr_matrix = df[numerical_features].corr('spearman')
    # mask the upper triangle:
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    # get the y_tick:
    x_tick = corr_matrix.index
    # replace the last y_tick label with an empty string:
    y_tick = [col.replace('limit_bal', '') for col in x_tick]

    # plot the correlation matrix:
    plt.figure(figsize=(20, 20))
    sns.heatmap(corr_matrix, annot=True, cbar=True, cmap='coolwarm', mask=mask,
                fmt='.2f', annot_kws={'size': 10}, linewidths=0.5,
                xticklabels=x_tick[:-1], yticklabels=y_tick)

    # Annotate the correlation matrix:
    plt.title('Correlation Matrix with Spearman Correlation')
    # Save the plot:
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_path, 'correlation_matrix.png'))
    plt.close()


def plot_categorical_feature_distribution(df: pd.DataFrame) -> None:
    """
    Plot the distribution of the categorical features in the dataset.
    @param df: pd.DataFrame: the dataframe with the categorical features.
    :return: None
    """
    for feature in df.columns:

        if df[feature].dtype == 'category':
            # get the labels for the x-axis and the mapping for the legend:
            mapping = category_mapping(feature)
            # legend labels, add the frequency of each category, keep the original order:
            labels = [f'{mapping[cat]} ({df[feature].value_counts()[cat]})'
                      for cat in df[feature].cat.categories]
            # plot the distribution:
            plt.figure(figsize=(10, 10))
            # order the categories by their frequency:
            sns.countplot(x=feature, data=df, palette='tab10', hue=df[feature])
            # map the x-axis labels:
            plt.xticks(ticks=range(len(mapping)), labels=mapping.values(), rotation=15)
            # set the legend:
            plt.legend(labels=labels, loc='upper right', title=feature.capitalize())
            # annotate the plot:
            plt.title(f'Distribution of {feature.capitalize()}')
            plt.xlabel(feature.capitalize())
            plt.ylabel('Count')
            # Save the plot:
            plot_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(Path(plot_path, f'{feature}_distribution.png'))
            plt.close()


def plot_numerical_feature_distribution(df: pd.DataFrame, log_transform: bool = False) -> None:
    """
    Plot the distribution of the numerical features in the dataset.
    @param df: pd.DataFrame: the dataframe with the numerical features.
    @param log_transform: bool: whether to apply a log transformation to the numerical features.
    :return: None
    """
    for feature in df.columns:
        if df[feature].dtype != 'category' and feature != 'id':

            if log_transform:
                # apply a log transformation to the feature avoiding the 0 values:
                df[feature] = df[feature].apply(lambda x: np.log(x + 1))
            plt.figure(figsize=(10, 10))
            sns.histplot(x=feature, data=df, kde=True, bins='auto')
            plt.title(f'Distribution of {feature.capitalize()}')
            plt.xlabel(feature.capitalize())
            plt.ylabel('Count')
            plot_path.mkdir(parents=True, exist_ok=True)

            if log_transform:
                plt.savefig(Path(plot_path, f'{feature}_distribution_log_transformed.png'))
            else:
                plt.savefig(Path(plot_path, f'{feature}_distribution.png'))

            plt.close()


def plot_numerical_correlation_with_target(df: pd.DataFrame, target: str) -> None:
    """
    Plot the correlation of the numerical features with the target.
    @param df: pd.DataFrame: the dataframe with the numerical features.
    @param target: str: the target feature.
    :return: None
    """
    # remove the id and the target:
    features = df.drop([target], axis=1)
    # compute the correlation matrix on the numerical features:
    corr_matrix = features.corrwith(df[target], method='spearman', numeric_only=True)
    # plot the correlation matrix:
    plt.figure(figsize=(15, 10))
    sns.barplot(x=corr_matrix.index, y=corr_matrix.values, palette='coolwarm')
    # Annotate the correlation matrix:
    plt.title(f'Spearman correlation of defaulting with the numerical features')
    plt.xlabel('Features')
    plt.ylabel('Spearman Correlation')
    # rotate the x-axis labels and decrease the font size:
    plt.xticks(rotation=15, fontsize=10)
    # Save the plot:
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_path, f'correlation_with_{target}.png'))
    plt.close()


def plot_categorical_correlation_with_target(df: pd.DataFrame, target: str) -> None:
    """
    Plot the correlation o the categorical features with the target.
    @param df: pd.DataFrame: the dataframe with the categorical features.
    @param target: str: the target feature.
    :return:
    """
    # Todo: Davide, Fabio, is there a better way to do this?
    # select only the categorical features:
    categorical_features = df.select_dtypes(include='category').columns
    # remove the target:
    categorical_features = categorical_features.drop(target)
    nominal_features = [feature for feature in categorical_features if not feature.startswith('pay_stat_')]
    # encode the nominal features, the pay_stat features are ordinal and already encoded:
    nominal_features = pd.get_dummies(df[nominal_features])
    # add the ordinal features:
    ordinal_features = df[[feature for feature in categorical_features if feature.startswith('pay_stat_')]]
    # concatenate the nominal and ordinal features:
    categorical_features = pd.concat([nominal_features, ordinal_features], axis=1)
    # compute the correlation matrix on the categorical features:
    corr_matrix = categorical_features.corrwith(df[target], method='pearson', numeric_only=False)
    # plot the correlation matrix:
    plt.figure(figsize=(20, 20))
    sns.barplot(x=corr_matrix.index, y=corr_matrix.values, palette='coolwarm')
    # Annotate the correlation matrix:
    plt.title(f'Pearson correlation of one hot encoded categorical features with the target')
    plt.xlabel('Features')
    plt.ylabel('Pearson Correlation')
    # rotate the x-axis labels and decrease the font size:
    plt.xticks(rotation=90, fontsize=8)
    # Save the plot:
    plot_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(Path(plot_path, f'correlation_with_{target}_one_hot_encoded.png'))
    plt.close()


@measure_time
def eda_main() -> None:
    """
    Main function.
    :return: None. Plot the correlation matrix and the distribution of the categorical features.
    """
    # Load the dataset:
    if not csv_file_path.exists():
        raise FileNotFoundError(f'Invalid data path: {csv_file_path}')

    df = pd.read_csv(csv_file_path)

    # Name of the categorical features from the dataset information:
    categorical_features: list[str] = ["gender", "education", "marriage", "pay_stat_sep", "pay_stat_aug",
                                       "pay_stat_jul",
                                       "pay_stat_jun", "pay_stat_may", "pay_stat_apr", "default"]
    # Assign the categorical features:
    df[categorical_features] = df[categorical_features].astype('category')

    # Plot the correlation matrix:
    plot_correlation_matrix(df)

    # Plot the distribution of the categorical features, target included:
    plot_categorical_feature_distribution(df)

    # Plot the distribution of the numerical features:
    plot_numerical_feature_distribution(df, log_transform=True)

    # Plot the correlation of the numerical features with the target:
    plot_numerical_correlation_with_target(df, target='default')

    # Plot the correlation of the categorical features with the target:
    plot_categorical_correlation_with_target(df, target='default')


# Driver:
if __name__ == '__main__':
    eda_main()

    """
    None of the features is very highly correlated with the target, which implies a good model
    will require a good combination of features.
    
    Limit balance seems relatively important to predict defaulting, the higher the limit balance 
    the lower the probability of defaulting.
    
    The payment status, in praticular in september, seems to be important to predict defaulting. The 
    other payment status features are also relevant.
    
    Paid amount features are also noteworthy, again with a recency bias (the more recent the higher
    the importance).
    
    The other features are not as important from the initial analysis, hopefully combining them with
    the other features will improve the model performance.
    """

