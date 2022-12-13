# The simple models like decision trees, knn and logistic regression
# suffer from multicollinearity within this dataset, so we simplified the dataset
# The search was done with grid search,
# see: https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html for more info.
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

# Feature selection:
from feature_selection.remove_correlated_features import simplify_dataset

from auxiliary.method_timer import measure_time
# Global variables:
from config import final_train_under_csv_path, final_val_under_csv_path, other_models_tuned_results_path


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
                            'min_samples_split': [2, 3, 4, 5],
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


@measure_time
def simple_models_main() -> None:
    """
    This function is the main function of the script.
    :return: None
    """
    # Simplify the dataset, our quick feature selection:
    x_train, y_train, x_val, y_val = simplify_dataset(final_train_under_csv_path, final_val_under_csv_path)

    # Tune the models:
    tuned_models = tuner(x_train, y_train)

    # Evaluate the models:
    evaluate_models(x_train, y_train, x_val, y_val, tuned_models)


# Driver:
if __name__ == '__main__':
    simple_models_main()

