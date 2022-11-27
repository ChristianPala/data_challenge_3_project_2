# Data manipulation:
import pandas as pd

# Tuning
import optuna
from optuna.trial import TrialState, Trial
from tuning.balanced_neural_network import fit_model, predict_model, evaluate_model

# Modelling:
from keras.models import Model, Sequential
from keras.layers import Dense, Layer
from pathlib import Path
from modelling.neural_network import create_model_with_layers
from modelling.train_test_validation_split import split_data

# Global variables:
from config import scaled_datasets_path

EPOCHS = 1


def generate_model(trial: Trial) -> Model:
    """
    Generates a model with a number of levels and an optimizer chosen by Optuna.
    :return: The model with the hyperparameters tuned by Optuna
    """
    
    # Generate layers between 5 to 8 layers according to optuna's trial
    layers_count = trial.suggest_int("Layers Count", 5, 8)
    layers = suggest_layers(trial, layers_count)

    # Optuna chooses an optimizer
    optimizer = trial.suggest_categorical("optimizer", ["adam"])

    # Define the base model and the input dimension
    # input_dim = trial.suggest_int("input_dim", 20, 50)
    m = Sequential()
    model = create_model_with_layers(m, layers=layers, optimizer=optimizer)

    return model

def suggest_layers(trial: Trial, count) -> list[Layer]:
    """
    Creates layers using Optuna's hyperparameter tuning's suggestions. 
    Always generates count+1 number of layers, where the final layer has one 
    neuron and uses sigmoid as the activation function.
    @param trial: The current Optuna trial
    @param count: the number of layers which will be generated
    :return: a list of count+1 layers
    """
    layers = []
    for i in range(count):
        neurons = trial.suggest_int("layer_{}".format(i), 10, 200)
        if i == 0:
            layers.append(Dense(neurons, activation='relu', input_dim=23))
        else:
            layers.append(Dense(neurons, activation='relu'))
    layers.append(Dense(1, activation='sigmoid'))
        
    return layers


def objective(trial: Trial) -> float:
    """
    The function which is used by Optuna to optimize the chosen evaluation metric
    @param trial: The current Optuna trial
    """

    for epoch in range(EPOCHS):

        model = generate_model(trial)

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
    a detailed overview of the best neural network.
    """
    # Create Optuna Study which tries to maximize the value of our objective function (in this case it is maximizing accuracy)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=5, show_progress_bar=True)
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

# Todo: change the model builder if you get better results.
# parameters: {'n_layers': 3, 'n_units_l0': 508, 'activation': 'relu',
# 'dropout': 0.1571962679833076, 'n_units_l1': 164, 'n_units_l2': 512,
# 'optimizer': 'adam', 'epochs': 100, 'batch_size': 126}.
# Best is trial 20 with value: 0.8183366298675537 accuracy, since the dataset is balanced via under-sampling.
    

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