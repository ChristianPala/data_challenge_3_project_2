# Auxiliary library to tune the hyperparameters of the support vector classifier
# Libraries:
# Data manipulation:
from __future__ import annotations
import pandas as pd
# Modelling:
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import final_train_csv_path, final_val_csv_path
from sklearn.svm import SVC

# Type hints:
from typing import Dict, Any


def objective(space, x_train: pd.DataFrame, y_train: pd.DataFrame, x_test: pd.DataFrame, y_test: pd.DataFrame,
              cross_validation: int = 5, fast: bool = False) -> Dict[str, int | Any]:
    """
    Objective function to be minimized.
    @param space: the hyperparameters to be tuned
    @param x_train: the training data
    @param y_train: the training labels
    @param x_test: the test data
    @param y_test: the test labels
    @param cross_validation: the number of folds for cross-validation
    @param fast: if True, the model is trained with minimal cross-validation
    :return: the f1 score as a loss metric
    """
    model = SVC(C=space['C'],
                kernel=space['kernel'],
                degree=space['degree'],
                random_state=42)

    # define the model evaluation data
    X = pd.concat([x_train, x_test])
    y = pd.concat([y_train, y_test])

    # since we have an imbalanced dataset, we need to use stratified k-fold cross-validation:
    cv = StratifiedKFold(n_splits=cross_validation, shuffle=True, random_state=42)

    if fast:
        cv = 2

    # since we are interested in churners, the positive class, the f1 is a good metric:
    metric = make_scorer(f1_score)

    # evaluate the model:
    f1 = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1).mean()

    # return the loss, 1 - f1 score since we want to maximize the f1 score:
    return {'loss': 1 - f1, 'status': STATUS_OK}


def tuner(x_train: pd.DataFrame, y_train: pd.DataFrame,
          x_test: pd.DataFrame, y_test: pd.DataFrame, max_evaluations: int = 100,
          cross_validation: int = 5, fast: bool = False) -> dict:
    """
    Tune the support vector classifier hyperparameters.
    @param x_train: the training data
    @param y_train: the training labels
    @param x_test: the test data
    @param y_test: the test labels
    @param max_evaluations: the maximum number of evaluations
    @param cross_validation: the number of folds for cross-validation
    @param fast: if True, the model is trained with minimal cross-validation
    :return: the best hyperparameters
    """

    # define the search space, choose the parameters to tune:
    space = {
        'C': hp.uniform('C', 10, 1000),
        'kernel': hp.choice('kernel', ['linear', 'rbf', 'sigmoid', 'poly']),
        'degree': hp.choice('degree', [2, 3, 4, 5, 6]),
    }

    # define the trials object:
    trials = Trials()

    # run the optimization:
    best = fmin(fn=lambda search_space: objective(search_space, x_train, y_train, x_test, y_test,
                                                  cross_validation, fast),
                space=space, algo=tpe.suggest, max_evals=max_evaluations, trials=trials)

    # filter out the parameters that are 0:
    best = {k: v for k, v in best.items() if v != 0}

    return best


def main() -> None:
    """
    Main function.
    @return: None. The best hyperparameters are printed.
    """
    # load the data:
    training = pd.read_csv(final_train_csv_path)
    validation = pd.read_csv(final_val_csv_path)

    # split the data into features and labels:
    x_train = training.drop('default', axis=1)
    y_train = training['default']
    x_test = validation.drop('default', axis=1)
    y_test = validation['default']

    # tune the model:
    best = tuner(x_train, y_train, x_test, y_test, max_evaluations=100, cross_validation=5, fast=False)
    print(best)

    # train the model with the best parameters:
    model = SVC(**best, random_state=42)
    model.fit(x_train, y_train)

    # train the model:
    model.fit(x_train, y_train)

    # evaluate the model:
    y_pred = model.predict(x_test)
    print(f1_score(y_test, y_pred))


if __name__ == '__main__':
    main()
