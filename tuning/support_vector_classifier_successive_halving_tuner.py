# Library to tune the support vector classifier model with successive halving.
# See https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.HalvingRandomSearchCV.html
# Libraries:
# Data manipulation:
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
# Import datasets:
from tuning.gradient_booster_successive_halving_tuner import create_csv_list
# Hyperparameter tuning:
# experimental tuning with halving random search cv:
from sklearn.experimental import enable_halving_search_cv  # NOQA
from sklearn.model_selection import HalvingRandomSearchCV, StratifiedKFold  # NOQA
# Modeling and metrics:
from sklearn.svm import SVC
from sklearn.metrics import f1_score
# Time:
from auxiliary.method_timer import measure_time
from tqdm import tqdm

# Global variables:
from config import other_models_tuned_results_path

# Ensure the directory exists:
other_models_tuned_results_path.mkdir(parents=True, exist_ok=True)


def create_model() -> SVC:
    """
    This function creates the model to be tuned.
    :return: model: the model to be tuned.
    """
    # Create the model:
    model = SVC(random_state=42, probability=True)

    return model


def tune_model(training_csv_list: List[Path]) -> list[SVC]:
    """
    This function tunes the model on the balanced data.
    @param training_csv_list: the list of paths to the csv files in the balanced folder.
    @return: the tuned models
    """
    # define the parameter space:
    param_space = {
        'C': [5, 6, 7, 8, 9, 10, 100, 1000],
        'kernel': ['rbf', 'poly', 'sigmoid'],
        'degree': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'gamma': ['scale', 'auto', 10 ** -5, 10 ** -4, 10 ** -3],
    }

    # create the model:
    model = create_model()
    tuned_models = []

    # tune the model:
    for csv_path in tqdm(training_csv_list):
        # read the data:
        data = pd.read_csv(csv_path)
        # create the X and y:
        X = data.drop(columns=['default'])
        y = data['default']

        # create the cv:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # tune the model:
        halving_random_search_cv = HalvingRandomSearchCV(
            model,
            param_space,
            cv=cv,
            n_jobs=-1,
            random_state=42,
            verbose=1  # verbose=1 to print the progress, since it's pretty nice.
            # for larger datasets like svm_smote, aggressive_elimination=True makes sense.
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
        # get the balancing method, one folder up from the file name:
        balancing_method = csv.parents[1].name
        # read the data:
        validation = pd.read_csv(csv)
        # split the data:
        x = validation.drop('default', axis=1)
        y = validation['default']
        # evaluate the tuned model:
        y_pred = tuned_model.predict(x)
        y_pred = np.where(y_pred > 0.5, 1, 0)
        # calculate the f1 score:
        f1 = f1_score(y, y_pred)
        # save the results:
        results.append([tuned_model, f1, balancing_method])
    # create the dataframe:
    results = pd.DataFrame(results, columns=['model', 'f1 score', 'balancing method'])

    # save the results:
    results.to_csv(Path(other_models_tuned_results_path, "svc_tuned.csv"), index=False)


@measure_time
def svc_main() -> None:
    """
    This function performs the tuning of the gradient booster model with successive halving.
    :return: None
    """
    # create the list of paths to the csv files in the balanced folder:
    training_csv_list, validation_csv_list = create_csv_list(added_path=Path('undersampled'))
    # tune the model:
    tuned_models = tune_model(training_csv_list)
    # evaluate the tuned models:
    evaluate_models(tuned_models, validation_csv_list)


if __name__ == '__main__':
    svc_main()


