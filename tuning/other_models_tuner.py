# Data manipulation:
from pathlib import Path
import pandas as pd

# Modelling:
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from modelling.model_evaluator import  evaluate_model
from modelling.train_test_validation_split import split_data
from modelling.knn_logreg_naiveb_svc import fit_model, evaluate_model, predict_model

# Tuning
import optuna
from optuna.trial import TrialState, Trial

# Global variables:
from config import scaled_datasets_path

def other_model_tuner(trial:Trial, model) -> BaseEstimator:
    if type(model) == SVC:
        c_value = trial.suggest_float("C", 1.0, 5.0)
        kernel = trial.suggest_categorical("kernel", ["linear", "poly", "rbf", "sigmoid", "precomputed"])
        degree = trial.suggest_int("degree", 2, 5)
        gamma =  trial.suggest_categorical("gamma", ["scale", "auto"])
        return SVC(c_value, kernel, degree, gamma)
    elif type(model) == KNeighborsClassifier:
        n_neighbors = trial.suggest_int("n_neighbors", 3, 10)
        return KNeighborsClassifier(n_neighbors)
    elif type(model) == LogisticRegression:
        penalty =  trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", "none"])
        c_value = trial.suggest_float("C", 1.0, 5.0)
        return LogisticRegression(penalty=penalty, C=c_value)
    elif type(model) == GaussianNB:
        var_smoothing = trial.suggest_float("var_smoothing", 1e-10, 1e-8) 
        return GaussianNB(var_smoothing=var_smoothing)

    

def objective(trial: Trial):
    """
    The function which is used by Optuna to optimize the chosen evaluation metric
    @param trial: The current Optuna trial
    """
    
    for epoch in range(1):
        estimator = KNeighborsClassifier()
        model = other_model_tuner(trial, estimator)

        csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))
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