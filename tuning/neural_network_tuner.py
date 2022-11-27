# Tune the simple neural network model with Optuna:
# The best dataset in the balanced folder was the minmax scaler scaling and most frequent imputation dataset in
# the undersampled folder.

# Libraries:
# Modelling:
import optuna
from keras import Sequential
from keras.backend import clear_session
from keras.layers import Dense, Dropout
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.model_selection import StratifiedKFold

# Data manipulation:
from pathlib import Path
import pandas as pd

# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
from config import balanced_datasets_path, neural_networks_balanced_results_path


# Functions:
def load_best_dataset(path: Path = Path(balanced_datasets_path, "undersampled",
                                        "minmax_scaler_scaling_most_frequent_imputation")) \
        -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    This function loads the best dataset from the balanced folder.
    :return: pd.DataFrame: the best dataset.
    """
    # load the dataset:
    train = pd.read_csv(Path(path, "train.csv"))
    val = pd.read_csv(Path(path, "val.csv"))
    # split the dataset into features and target:
    x_train = train.drop('default', axis=1)
    y_train = train['default']
    x_val = val.drop('default', axis=1)
    y_val = val['default']

    return x_train, y_train, x_val, y_val


def objective(trial):
    clear_session()
    # load the best dataset:
    x_train, y_train, x_val, y_val = load_best_dataset()

    # define the model:
    model = Sequential()
    for i in range(trial.suggest_int("n_layers", 1, 3)):
        model.add(Dense(units=trial.suggest_int("n_units_l{}".format(i), 4, 512),
                        activation=trial.suggest_categorical("activation", ["relu", "elu",
                                                                            "selu", "tanh"])))
        # add dropout:
        model.add(Dropout(trial.suggest_float("dropout", 0.0, 0.5)))

    # add the output layer:
    model.add(Dense(1, activation="sigmoid"))

    # use f1 score with cross validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, test_index in cv.split(x_train, y_train):
        # split the data:
        x_train_cv, x_test_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]
        # compile the model:
        model.compile(optimizer=trial.suggest_categorical("optimizer", ["adam", "sgd", "rmsprop"]),
                      loss="binary_crossentropy", metrics=["accuracy"])
        # fit the model:
        model.fit(x_train_cv, y_train_cv, epochs=trial.suggest_int("epochs", 10, 100),
                  batch_size=trial.suggest_int("batch_size", 8, 128), verbose=0)

        # evaluate the model:
        score = model.evaluate(x_test_cv, y_test_cv, verbose=0)

        # append the score:
        cv_scores.append(score[1])

    # return the mean of the scores:
    return sum(cv_scores) / len(cv_scores)


# Main:
@measure_time
def main():
    # set the sampler:
    sampler = TPESampler(seed=42)
    # set the pruner:
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    # set the study:
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    # run the study:
    study.optimize(objective, n_trials=100, n_jobs=-1)
    # save the results:
    study.trials_dataframe().to_csv(Path(neural_networks_balanced_results_path, "neural_network_tuner.csv"))


if __name__ == "__main__":
    main()


