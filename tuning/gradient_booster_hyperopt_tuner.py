# Auxiliary library to tune the gradient booster model.
# Note: the code was adapted from the following the xgbost tuner of project 1 by Christian Berchtold and Christian Pala.
# Library used: hyperopt, see: http://hyperopt.github.io/hyperopt/
# Libraries:
# Data manipulation:
from __future__ import annotations
import pandas as pd
# Modelling:
from hyperopt import hp, Trials, fmin, STATUS_OK, tpe
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import cross_val_score, StratifiedKFold
from config import final_train_under_csv_path, final_val_under_csv_path
from sklearn.ensemble import GradientBoostingClassifier
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
    model = GradientBoostingClassifier(n_estimators=space['n_estimators'],
                                       loss=space['loss'],
                                       max_depth=space['max_depth'],
                                       learning_rate=space['learning_rate'],
                                       subsample=space['subsample'],
                                       criterion=space['criterion'],
                                       min_samples_split=space['min_samples_split'],
                                       min_samples_leaf=space['min_samples_leaf'],
                                       min_weight_fraction_leaf=space['min_weight_fraction_leaf'],
                                       max_leaf_nodes=space['max_leaf_nodes'],
                                       min_impurity_decrease=space['min_impurity_decrease'],
                                       random_state=42)

    # define the model evaluation data
    X = pd.concat([x_train, x_test])
    y = pd.concat([y_train, y_test])

    # since we have an imbalanced dataset, we need to use stratified k-fold cross-validation:
    cv = StratifiedKFold(n_splits=cross_validation, shuffle=True, random_state=42)

    if fast:
        cv = 2

    # define the evaluation metric:
    metric = make_scorer(f1_score)

    # evaluate the model:
    f1 = cross_val_score(model, X, y, scoring=metric, cv=cv, n_jobs=-1).mean()

    # return the loss metric:
    return {'loss': 1 - f1, 'status': STATUS_OK}


def tuner(x_train: pd.DataFrame, y_train: pd.DataFrame,
          x_test: pd.DataFrame, y_test: pd.DataFrame, max_evaluations: int = 100,
          cross_validation: int = 5, fast: bool = False) -> dict:
    """
    Tune the gradient boosting model.
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
        'loss': hp.choice('loss', ["log_loss", "exponential"]),
        'max_depth': hp.choice('max_depth', range(1, 10)),
        'learning_rate': hp.uniform('learning_rate', 10 ** -5, 0.5),
        'subsample': hp.uniform('subsample', 0.5, 1),
        'criterion': hp.choice('criterion', ['friedman_mse', 'squared_error']),
        'min_samples_split': hp.choice('min_samples_split', range(2, 10)),
        'min_samples_leaf': hp.choice('min_samples_leaf', range(1, 10)),
        'min_weight_fraction_leaf': hp.uniform('min_weight_fraction_leaf', 0, 0.5),
        'max_leaf_nodes': hp.choice('max_leaf_nodes', range(2, 10)),
        'min_impurity_decrease': hp.uniform('min_impurity_decrease', 0, 0.5)
    }

    # define the trials object:
    trials = Trials()

    # run the optimization:
    best = fmin(fn=lambda search_space: objective(search_space, x_train, y_train, x_test, y_test,
                                                  cross_validation, fast),
                space=space, algo=tpe.suggest, max_evals=max_evaluations, trials=trials)

    # filter out the parameters that are 0 or non-numerical, to avoid halting the program:
    best = {k: v for k, v in best.items() if v != 0}
    best = {k: v for k, v in best.items() if isinstance(v, int) or isinstance(v, float)}

    return best


def main() -> None:
    """
    Main function.
    @return: None. The best hyperparameters are printed.
    """
    # load the data:
    training = pd.read_csv(final_train_under_csv_path)
    validation = pd.read_csv(final_val_under_csv_path)

    # split the data into features and labels:
    x_train = training.drop('default', axis=1)
    y_train = training['default']
    x_test = validation.drop('default', axis=1)
    y_test = validation['default']

    # tune the model:
    best = tuner(x_train, y_train, x_test, y_test, max_evaluations=100, cross_validation=5, fast=False)

    # print the best hyperparameters:
    # check the hyperparameters are implemented correctly below:
    print(best)

    # train the model with the best parameters:
    model = GradientBoostingClassifier(random_state=42, **best)

    # train the model:
    model.fit(x_train, y_train)

    # evaluate the model on the validation set:
    y_pred = model.predict(x_test)
    print(f"F1: {f1_score(y_test, y_pred)}")


if __name__ == '__main__':
    main()
