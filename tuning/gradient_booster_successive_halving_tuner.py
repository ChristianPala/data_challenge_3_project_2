# Library to perform the tuning of the gradient booster model with successive halving.

# Libraries:
# Data manipulation:
from pathlib import Path
from typing import List, Tuple

import pandas as pd
# Trees
from sklearn.ensemble import GradientBoostingClassifier
# experimental tuning with halving random search cv:
from sklearn.experimental import enable_halving_search_cv  # NOQA
from sklearn.metrics import f1_score
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold  # NOQA
from tqdm import tqdm
# Time:
from auxiliary.method_timer import measure_time
# Global variables:
from config import balanced_datasets_path, trees_balanced_results_path

# ensure the trees_balanced_results_path exists:
Path(trees_balanced_results_path).mkdir(parents=True, exist_ok=True)


def create_model() -> GradientBoostingClassifier:
    """
    This function creates the model to be tuned.
    :return: model: the model to be tuned.
    """
    # Create the model:
    model = GradientBoostingClassifier(random_state=42)

    return model


def create_csv_list(added_path: Path = None) -> Tuple[list[Path], list[Path]]:
    """
    This function creates a list of paths to the csv files in the balanced folder.
    :return: list[Path]: a list of paths to the csv files in the balanced folder.
    """
    # if a sub folder is specified:
    path = balanced_datasets_path if added_path is None else balanced_datasets_path / added_path

    # create the list of paths:
    training_csv_list = list(Path(path).rglob("final_training.csv"))
    validation_csv_list = list(Path(path).rglob("final_validation.csv"))

    return training_csv_list, validation_csv_list


def tune_model(training_csv_list: List[Path]) -> list[GradientBoostingClassifier]:
    """
    This function tunes the model on the balanced data.
    @param training_csv_list: the list of paths to the csv files in the balanced folder.
    @return: the tuned models
    """
    # define the parameter space:
    param_space = {
        'n_estimators': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        'learning_rate': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'max_depth': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_split': [2, 3, 4, 5, 6, 7, 8, 9, 10],
        'min_samples_leaf': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
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
    for csv in training_csv_list:
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


def evaluate_models(tuned_models: List[GradientBoostingClassifier], validation_csv_list: List[Path]) -> None:
    """
    This function evaluates the tuned models on the balanced data.
    @param tuned_models: the tuned models.
    @param validation_csv_list: the list of paths to the validation csv files in the balanced folder.
    @return: None
    """

    # create the list of results:
    results = []
    # evaluate the tuned models on each dataset:
    for tuned_model, csv in tqdm(zip(tuned_models, validation_csv_list), total=len(validation_csv_list),
                                 desc='Evaluating models', unit='model', colour='green'):
        # read the data:
        validation = pd.read_csv(csv)
        # split the data:
        x = validation.drop('default', axis=1)
        y = validation['default']
        # evaluate the tuned model:
        y_pred = tuned_model.predict(x)
        # calculate f1 score:
        f1 = f1_score(y, y_pred)
        results.append([tuned_model, f1])
    # create the dataframe:
    results = pd.DataFrame(results, columns=['model', 'score'])
    # save the results:
    results.to_csv(Path(trees_balanced_results_path, "gradient_booster_successive_halving_tuner_results.csv"),
                   index=False)
    # select the best model:
    best_model = results.loc[results['score'].idxmax(), 'model']
    # print the best model parameters:
    print(best_model.get_params())
    """
    {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.2, 
    'loss': 'log_loss', 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': None, 
    'min_impurity_decrease': 0.0, 'min_samples_leaf': 5, 'min_samples_split': 5, 
    'min_weight_fraction_leaf': 0.0, 'n_estimators': 800, 'n_iter_no_change': None, '
    random_state': 42, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
    """


@measure_time
def gb_main() -> None:
    """
    This function performs the tuning of the gradient booster model with successive halving.
    :return: None
    """
    # create the list of paths to the csv files in the balanced folder:
    training_csv_list, validation_csv_list = create_csv_list(added_path=Path('svm_smote'))
    # tune the model:
    tuned_models = tune_model(training_csv_list)
    # evaluate the tuned models:
    evaluate_models(tuned_models, validation_csv_list)


if __name__ == '__main__':
    gb_main()
