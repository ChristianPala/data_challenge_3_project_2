# Library to perform the tuning of the gradient booster model with successive halving.

# Libraries:
# Data manipulation:
from pathlib import Path
from typing import List

import pandas as pd
# Trees
from sklearn.ensemble import GradientBoostingClassifier
# experimental tuning with halvingrandomsearchcv:
from sklearn.experimental import enable_halving_search_cv  # NOQA
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold
from tqdm import tqdm

from auxiliary.method_timer import measure_time
# Global variables:
from config import balanced_datasets_path, trees_balanced_results_path


def create_model() -> GradientBoostingClassifier:
    """
    This function creates the model to be tuned.
    :return: model: the model to be tuned.
    """
    # Create the model:
    model = GradientBoostingClassifier(random_state=42, max_features=1.0)

    return model


def create_csv_list() -> list[Path]:
    """
    This function creates a list of paths to the csv files in the balanced folder.
    :return: list[Path]: a list of paths to the csv files in the balanced folder.
    """
    # create the list of paths:
    csv_list = list(Path(balanced_datasets_path).rglob("*.csv"))

    return csv_list


def tune_model(csv_list: List[Path]) -> list[GradientBoostingClassifier]:
    """
    This function tunes the model on the balanced data.
    @param csv_list: the list of paths to the csv files in the balanced folder.
    @return: the tuned models
    """
    # define the parameter space:
    param_space = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'criterion': ['friedman_mse', 'squared_error']
    }

    # create the model:
    model = create_model()
    # create the cross validation object:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # create the tuning object:
    tuning = HalvingRandomSearchCV(model, param_space, cv=cv, n_jobs=-1, random_state=42, verbose=1,
                                   aggressive_elimination=True, factor=3)
    # create the list of tuned models:
    tuned_models = []
    # tune the model on each dataset:
    for csv in csv_list:
        # read the data:
        data = pd.read_csv(csv)
        # split the data:
        x = data.drop('default', axis=1)
        y = data['default']
        # tune the model:
        tuning.fit(x, y)
        # save the tuned model:
        tuned_models.append(tuning.best_estimator_)

    return tuned_models


def evaluate_models(tuned_models: List[GradientBoostingClassifier], csv_list: List[Path]) -> None:
    """
    This function evaluates the tuned models on the balanced data.
    @param tuned_models: the tuned models.
    @param csv_list: the list of paths to the csv files in the balanced folder.
    @return: None
    """

    # create the list of results:
    results = []
    # evaluate the tuned models on each dataset:
    for tuned_model, csv in tqdm(zip(tuned_models, csv_list), total=len(csv_list), desc='Evaluating models',
                                 unit='model', colour='green'):
        # read the data:
        data = pd.read_csv(csv + 'final_training.csv')
        # split the data:
        x = data.drop('class', axis=1)
        y = data['class']
        # evaluate the tuned model:
        score = tuned_model.score(x, y)
        # save the results:
        results.append([tuned_model, score])
    # create the dataframe:
    results = pd.DataFrame(results, columns=['model', 'score'])
    # save the results:
    results.to_csv(trees_balanced_results_path, index=False)
    # select the best model:
    best_model = results.loc[results['score'].idxmax(), 'model']
    # print the best model parameters:
    print(best_model.get_params())


@measure_time
def main() -> None:
    """
    This function performs the tuning of the gradient booster model with successive halving.
    :return: None
    """
    # create the list of paths to the csv files in the balanced folder:
    csv_list = create_csv_list()
    # tune the model:
    tuned_models = tune_model(csv_list)
    # evaluate the tuned models:
    evaluate_models(tuned_models, csv_list)


if __name__ == '__main__':
    main()





