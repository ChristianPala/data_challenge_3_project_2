# Data manipulation:
import pandas as pd
from pathlib import Path

# Modelling
import optuna
from optuna.trial import TrialState, Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
from keras.backend import clear_session
from modelling.model_evaluator import evaluate_model
from modelling.train_test_validation_split import split_data
from modelling.trees import fit_model, evaluate_model, predict_model



# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
from config import balanced_datasets_path, scaled_datasets_path, trees_tuned_results_path


def load_best_dataset(path: Path = Path(balanced_datasets_path, "undersampled",
                                        "minmax_scaler_scaling_most_frequent_imputation")) \
        -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    This function loads the best dataset from the balanced folder.
    :return: pd.DataFrame: the best dataset.
    """
    # load the dataset:
    train = pd.read_csv(Path(path, "final_training.csv"))
    val = pd.read_csv(Path(path, "final_validation.csv"))
    # split the dataset into features and target:
    x_train = train.drop('default', axis=1)
    y_train = train['default']
    x_val = val.drop('default', axis=1)
    y_val = val['default']

    return x_train, y_train, x_val, y_val


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
    clear_session()
    # load the best dataset:
    #x_train, y_train, x_val, y_val = load_best_dataset()

    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))
    df = pd.read_csv(csv_files[0])

    # split the data into train and test:
    x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)

    # define the model:
    estimator = RandomForestClassifier()
    model = generate_estimator(trial, estimator)

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
    study.trials_dataframe().to_csv(Path(trees_tuned_results_path, "trees_tuner.csv"))


if __name__ == '__main__':
    main()

