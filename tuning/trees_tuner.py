# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling
import optuna
from optuna.trial import TrialState, Trial
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from modelling.model_evaluator import evaluate_model
from modelling.train_test_validation_split import split_data
from modelling.trees import fit_model, evaluate_model, predict_model

# Timing:
from config import missing_values_path


def generate_estimator(trial: Trial, estimator: BaseEstimator):
    """
    Generates an estimator using Optuna's suggestions for parameter tuning
    @param trial: The current Optuna trial
    @param estimator: BaseEstimator: the model to be used for prediction.
    :return: the estimator with the tuned parameters
    """
    categorical: dict = {}
    numerical = []

    if type(estimator) == DecisionTreeClassifier:
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])
        splitter = trial.suggest_categorical("splitter", ["best", "random"])

        max_depth = trial.suggest_int("max_depth", 50, 200)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)

        return DecisionTreeClassifier(criterion=criterion, splitter=splitter, 
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
        
    elif type(estimator) == RandomForestClassifier:
        criterion = trial.suggest_categorical("criterion", ["gini", "entropy"])

        max_depth = trial.suggest_int("max_depth", 50, 200)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)
        return RandomForestClassifier(criterion=criterion, max_depth=max_depth, 
            min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
    
    elif type(estimator) == GradientBoostingClassifier:
        criterion = trial.suggest_categorical("criterion", ["friedman_mse", "squared_error", "mse"])
        learning_rate = trial.suggest_float("lr", 0.1, 10)
        n_estimators = trial.suggest_int("n_estimators", 100, 300)

        max_depth = trial.suggest_int("max_depth", 50, 200)
        min_samples_split = trial.suggest_int("min_samples_split", 2, 5)
        min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 3)

        return GradientBoostingClassifier(criterion=criterion, learning_rate=learning_rate, n_estimators=n_estimators,
            max_depth=max_depth, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)

    
    elif type(estimator) == XGBClassifier:
        return XGBClassifier()
    return  categorical, numerical


def objective(trial: Trial):
    """
    The function which is used by Optuna to optimize the chosen evaluation metric
    @param trial: The current Optuna trial
    """
    
    for epoch in range(1):
        estimator = RandomForestClassifier()
        model = generate_estimator(trial, estimator)

        csv_files: list[Path] = list(missing_values_path.glob('*.csv'))
        df = pd.read_csv(csv_files[0])

        # split the data into train and test:
        x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)

        # fit the model:
        model = fit_model(model, x_train, y_train)

        # predict the target values:
        y_pred = predict_model(model, x_val)

        # evaluate the model:
        evaluation_results = evaluate_model(y_val, y_pred)
        
        # Report the accuracy for the current trial
        accuracy = evaluation_results["accuracy"]
        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy

def tune():
    """
    Creates the Optuna study and prints the results for each trial. When the Optuna study is concluded, it prints
    a detailed overview of the best estimator.
    """
    # Create Optuna Study which tries to maximize the value of our objective function (in this case it is maximizing accuracy)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=True)
    
    # Take the aggregate results of the studies
    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    # Show the results
    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    # Show the best trial
    print("Best trial:")
    trial = study.best_trial

    # Show the value of the best trial
    print("  Value: ", trial.value)

    # Show the parameters of the model with the best trial
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

if __name__ == '__main__':
    tune()