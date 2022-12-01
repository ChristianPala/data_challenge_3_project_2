# Data manipulation:
from pathlib import Path
import pandas as pd

# Modelling:
from sklearn.base import BaseEstimator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from keras.backend import clear_session
from modelling.model_evaluator import  evaluate_model
from modelling.train_test_validation_split import split_data
from modelling.knn_logreg_naiveb_svc import fit_model, evaluate_model, predict_model

# Tuning
import optuna
from optuna.trial import TrialState, Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
from config import other_models_tuned_results_path, scaled_datasets_path

def generate_model(trial:Trial, model) -> BaseEstimator:
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
    clear_session()
    # load the best dataset:
    #x_train, y_train, x_val, y_val = load_best_dataset()

    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))
    df = pd.read_csv(csv_files[0])

    # split the data into train and test:
    x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)

    # define the model:
    estimator = KNeighborsClassifier()
    model = generate_model(trial, estimator)

    # use f1 score with cross validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, test_index in cv.split(x_train, y_train):
        # split the data:
        x_train_cv, x_test_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # fit the model:
        # fit the model:
        model = fit_model(model, x_train_cv, y_train_cv)

        # predict the target values:
        y_pred = predict_model(model, x_test_cv)

        # evaluate the model:
        score = evaluate_model(y_test_cv, y_pred)

        # append the score:
        cv_scores.append(score["accuracy"])
    
    trial.report(sum(cv_scores) / len(cv_scores), 1)

    # return the mean of the scores:
    return sum(cv_scores) / len(cv_scores)


@measure_time
def main():
    # set the sampler:
    sampler = TPESampler(seed=42)
    # set the pruner:
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    # set the study:
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    # run the study:
    study.optimize(objective, n_trials=2, n_jobs=1, show_progress_bar=True)

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

    # save the results:
    study.trials_dataframe().to_csv(Path(other_models_tuned_results_path, "other_models_tuner.csv"))

if __name__ == '__main__':
    main()