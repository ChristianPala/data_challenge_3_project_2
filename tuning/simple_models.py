# The simple models like decision trees, knn and logistic regression
# suffer from multicollinearity with this dataset, so we try them on a reduced dataset.
# Libraries:
# Data manipulation:
import pandas as pd
from pathlib import Path
# Type hinting:
from typing import List, Tuple

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
# Timing:
from tqdm import tqdm

# Global variables:
from config import final_train_csv_path, final_val_csv_path, other_models_tuned_results_path


def simplify_dataset() -> (pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame):
    """
    This function simplifies the dataset by removing the columns with the highest correlation.
    :return: None
    """
    # Load the data:
    train_df = pd.read_csv(final_train_csv_path)
    val_df = pd.read_csv(final_val_csv_path)

    x_train = train_df.drop(columns=['default'])
    y_train = train_df['default']

    x_val = val_df.drop(columns=['default'])
    y_val = val_df['default']

    # since we already created total pay_status_cumulative, we can remove the other pay_status columns:
    x_train = x_train.drop(columns=['pay_stat_sep', 'pay_stat_aug', 'pay_stat_jul', 'pay_stat_jun',
                                    'pay_stat_may', 'pay_stat_apr'])
    x_val = x_val.drop(columns=['pay_stat_sep', 'pay_stat_aug', 'pay_stat_jul', 'pay_stat_jun',
                                'pay_stat_may', 'pay_stat_apr'])

    # since we already created total_bill_amount, we can remove the other bill_amt columns:
    x_train = x_train.drop(columns=['bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun',
                                    'bill_amt_may', 'bill_amt_apr'])

    x_val = x_val.drop(columns=['bill_amt_sep', 'bill_amt_aug', 'bill_amt_jul', 'bill_amt_jun',
                                'bill_amt_may', 'bill_amt_apr'])

    # since we already created total_paid_amount, we can remove the other pay_amt columns:
    x_train = x_train.drop(columns=['pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun',
                                    'pay_amt_may', 'pay_amt_apr'])

    x_val = x_val.drop(columns=['pay_amt_sep', 'pay_amt_aug', 'pay_amt_jul', 'pay_amt_jun',
                                'pay_amt_may', 'pay_amt_apr'])

    return x_train, y_train, x_val, y_val


def create_models() -> List[Tuple[str, object]]:
    """
    This function creates the models to be tuned.
    :return: models: the list of models to be tuned.
    """
    # Create the models:
    models = [('decision_tree', DecisionTreeClassifier(random_state=42)),
              ('knn', KNeighborsClassifier()),
              ('logistic_regression', LogisticRegression(random_state=42))]

    return models


def tuner(x_train: pd.DataFrame, y_train: pd.DataFrame) -> List[Tuple[str, object]]:
    """
    This function tunes the models.
    :return: models: the list of tuned models.
    """
    # Create the models:
    models = create_models()

    # define the parameters to be tuned:
    decision_tree_params = {'max_depth': [3, 5, 7, 9, 11],
                            'min_samples_leaf': [1, 2, 3, 4, 5],
                            'min_samples_split': [1, 2, 3, 4, 5],
                            'criterion': ['gini', 'entropy']}

    knn_params = {'n_neighbors': [3, 5, 7, 9, 11, 13],
                  'weights': ['uniform', 'distance'],
                  'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']}

    logistic_regression_params = {'penalty': ['l1', 'l2'],
                                  'C': [3, 4, 5, 6, 7],
                                  'solver': ['liblinear', 'saga'],
                                  }

    # define the parameter grids:
    parameter_grids = [decision_tree_params, knn_params, logistic_regression_params]

    # create the tuned models:
    tuned_models = []
    for i in tqdm(range(len(models))):
        # create the grid search:
        grid_search = GridSearchCV(models[i][1], parameter_grids[i], scoring='f1', cv=5, n_jobs=-1)
        # fit the grid search:
        grid_search.fit(x_train, y_train)
        # print the best parameters:
        print(f'Best parameters for {models[i][0]}: {grid_search.best_params_}')
        # append the tuned model to the list:
        tuned_models.append((models[i][0], grid_search.best_estimator_))

    return tuned_models


def evaluate_models(x_train: pd.DataFrame, y_train: pd.Series, x_val: pd.DataFrame, y_val: pd.Series,
                    models: List[Tuple[str, object]]) -> None:
    """
    This function tunes the models.
    :param x_train: the training set.
    :param y_train: the training set labels.
    :param x_val: the validation set.
    :param y_val: the validation set labels.
    :param models: the list of models to be tuned.
    :return: None
    """
    # Create the results dataframe:
    results_df = pd.DataFrame(columns=['model', 'f1_score'])

    # Loop over the models:
    for model_name, model in tqdm(models):
        # Fit the model:
        model.fit(x_train, y_train)

        # Evaluate the model:
        y_pred = model.predict(x_val)

        # Calculate the f1 score:
        f1 = f1_score(y_val, y_pred)

        # concatenate the results:
        results_df = pd.concat([results_df, pd.DataFrame({'model': [model_name], 'f1_score': [f1]})])

    # Save the results:
    results_df.to_csv(Path(other_models_tuned_results_path, "simple_models.csv"), index=False)


def main() -> None:
    """
    This function is the main function of the script.
    :return: None
    """
    # Simplify the dataset:
    x_train, y_train, x_val, y_val = simplify_dataset()

    # Tune the models:
    tuned_models = tuner(x_train, y_train)

    # Evaluate the models:
    evaluate_models(x_train, y_train, x_val, y_val, tuned_models)


# Driver:
if __name__ == '__main__':
    main()

