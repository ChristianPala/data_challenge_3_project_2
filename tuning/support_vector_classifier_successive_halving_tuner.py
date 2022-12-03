# Library to tune the support vector classifier model with successive halving.
# Libraries:
# Data manipulation:
from pathlib import Path
from typing import List
import pandas as pd
from tuning.gradient_booster_successive_halving_tuner import create_csv_list

# Hyperparameter tuning:
from sklearn.svm import SVC
# experimental tuning with halving random search cv:
from sklearn.experimental import enable_halving_search_cv  # NOQA
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold  # NOQA

# Time:
from auxiliary.method_timer import measure_time
from tqdm import tqdm
# Global variables:
from config import other_models_tuned_results_path


def create_model() -> SVC:
    """
    This function creates the model to be tuned.
    :return: model: the model to be tuned.
    """
    # Create the model:
    model = SVC(random_state=42)

    return model


def tune_model(training_csv_list: List[Path]) -> list[SVC]:
    """
    This function tunes the model on the balanced data.
    @param training_csv_list: the list of paths to the csv files in the balanced folder.
    @return: the tuned models
    """
    # define the parameter space:
    param_space = {
        'C': [0.1, 1, 10, 100, 1000],
        'gamma': [1, 0.1, 0.01, 0.001, 0.0001],
        'kernel': ['rbf', 'poly', 'sigmoid']
    }

    # create the model:
    model = create_model()
    tuned_models = []

    # tune the model:
    for csv_path in tqdm(training_csv_list):
        # read the data:
        data = pd.read_csv(csv_path)
        # create the X and y:
        X = data.drop(columns=['label'])
        y = data['label']

        # create the cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # tune the model:
        halving_random_search_cv = HalvingRandomSearchCV(
            model,
            param_space,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=0
        )

        halving_random_search_cv.fit(X, y)

        # save the tuned model:
        tuned_models.append(halving_random_search_cv.best_estimator_)

    return tuned_models


def evaluate_models(tuned_models: List[SVC], validation_csv_list: List[Path]) -> None:
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
        score = tuned_model.score(x, y)
        # save the results:
        results.append([tuned_model, score])
    # create the dataframe:
    results = pd.DataFrame(results, columns=['model', 'score'])
    # save the results:
    results.to_csv(other_models_tuned_results_path, index=False)
    # select the best model:
    best_model = results.loc[results['score'].idxmax(), 'model']
    # print the best model parameters:
    print(best_model.get_params())


@measure_time
def svc_main() -> None:
    """
    This function performs the tuning of the gradient booster model with successive halving.
    :return: None
    """
    # create the list of paths to the csv files in the balanced folder:
    training_csv_list, validation_csv_list = create_csv_list()
    # tune the model:
    tuned_models = tune_model(training_csv_list)
    # evaluate the tuned models:
    evaluate_models(tuned_models, validation_csv_list)


if __name__ == '__main__':
    svc_main()


