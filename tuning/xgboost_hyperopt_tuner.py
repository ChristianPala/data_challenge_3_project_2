# Auxiliary library to tune the xgboost model, code from project 1 by Christian Berchtold and Christian Pala.
# Libraries:
# Data manipulation:
from __future__ import annotations
import pandas as pd
# Modelling:
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import final_train_csv_path, final_val_csv_path
from xgboost import XGBClassifier
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
    model = XGBClassifier(n_estimators=space['n_estimators'],
                          max_depth=space['max_depth'],
                          learning_rate=space['learning_rate'],
                          min_child_weight=space['min_child_weight'],
                          gamma=space['gamma'],
                          subsample=space['subsample'],
                          colsample_bytree=space['colsample_bytree'],
                          reg_alpha=space['reg_alpha'],
                          reg_lambda=space['reg_lambda'],
                          objective="binary:logistic",
                          eval_metric="logloss",
                          random_state=42,
                          n_jobs=-1)

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
    return {'loss': 1-f1, 'status': STATUS_OK}


def tuner(x_train: pd.DataFrame, y_train: pd.DataFrame,
          x_test: pd.DataFrame, y_test: pd.DataFrame, max_evaluations: int = 100,
          cross_validation: int = 5, fast: bool = False) -> dict:
    """
    Tune the xgboost model.
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
        'n_estimators': hp.choice('n_estimators', range(100, 1000)),
        'max_depth': hp.choice('max_depth', range(1, 20)),
        'learning_rate': hp.uniform('learning_rate', 0.01, 1),
        'min_child_weight': hp.choice('min_child_weight', range(1, 10)),
        'gamma': hp.uniform('gamma', 0.01, 1),
        'subsample': hp.uniform('subsample', 0.01, 1),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.01, 1),
        'reg_alpha': hp.uniform('reg_alpha', 0.01, 1),
        'reg_lambda': hp.uniform('reg_lambda', 0.01, 1)
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


def main():
    # load the data:
    training = pd.read_csv(final_train_csv_path)
    validation = pd.read_csv(final_val_csv_path)

    # split the data into features and labels:
    x_train = training.drop('default', axis=1)
    y_train = training['default']
    x_test = validation.drop('default', axis=1)
    y_test = validation['default']

    # tune the model:
    best = tuner(x_train, y_train, x_test, y_test, max_evaluations=20, cross_validation=5, fast=False)

    # print the best parameters:
    print(best)

    # train the model with the best parameters:
    model = XGBClassifier(**best, objective="binary:logistic", eval_metric="logloss", random_state=42, n_jobs=-1)

    # fit the model:
    model.fit(x_train, y_train)

    # evaluate the model:
    f1 = f1_score(y_test, model.predict(x_test))

    # print the f1 score:
    print(f1)


if __name__ == '__main__':
    main()

