# Data manipulation:
import pandas as pd
from pathlib import Path
# Tuning
import optuna
from optuna.trial import TrialState, Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
# Modelling:
from keras.models import Model, Sequential
from keras.layers import Dense, Layer, Dropout
from keras.backend import clear_session
from keras.optimizers import Adam, RMSprop, SGD, Optimizer
from sklearn.model_selection import StratifiedKFold
from modelling.model_evaluator import evaluate_model, save_evaluation_results
# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
from config import balanced_datasets_path, neural_tuned_results_path

# Tensorflow logging:
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# How many trials to allow the tuner to run, time efficiency vs accuracy
NUMBER_OF_TRIALS: int = 20


def load_best_dataset() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    This function loads the best dataset from the balanced folder.
    :return: pd.DataFrame: the best dataset.
    """
    # load the dataset:
    train = pd.read_csv(Path(balanced_datasets_path, "smote_enn", "robust_scaler_scaling_drop", "final_training.csv"))
    val = pd.read_csv(Path(balanced_datasets_path, "smote_enn", "robust_scaler_scaling_drop", "final_validation.csv"))
    # split the dataset into features and target:
    x_train = train.drop('default', axis=1)
    y_train = train['default']
    x_val = val.drop('default', axis=1)
    y_val = val['default']

    return x_train, y_train, x_val, y_val


def create_model_with_layers(model: Model, layers: list[Layer], dropout: float = 0.1,
                             optimizer: Optimizer = Adam(), loss: str = 'binary_crossentropy',
                             metrics: list[str] = ['accuracy']) -> Model:
    """
    Creates a model with the given layers, optimizer, loss function and metrics.
    @param model: keras.Model: the model to add the layers to
    @param layers: keras.Layer: the layers to add to the model
    @param dropout: keras.Dropout: the dropout layer to add to the model
    @param optimizer: keras.Optimizer: the optimizer to use
    @param loss: keras.Loss: the loss function to use
    @param metrics: keras.Metrics: the metrics to use
    @param lr: float: the learning rate to use
    @return: keras.Model: the model with the given layers, optimizer, loss function and metrics.
    """
    compiled_model = model
    for i in range(len(layers)):
        compiled_model.add(layers[i])
        if i < len(layers) - 1:
            compiled_model.add(Dropout(dropout))

    compiled_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return compiled_model


def get_optimizer(choice: str, learning_rate: float = 0.001) -> Optimizer:
    """
    Returns an optimizer based on the choice and learning rate, auxiliary function for generate_model.
    @param choice: str: the optimizer to use, as a string
    @param learning_rate: float: the learning rate to use
    @return:
    """
    if choice == "adam":
        return Adam(learning_rate=learning_rate)
    elif choice == "rmsprop":
        return RMSprop(learning_rate=learning_rate)
    # else use SGD:
    return SGD(learning_rate=learning_rate)


def generate_model(trial: Trial) -> Model:
    """
    Generates a model with a number of levels and an optimizer chosen by Optuna.
    @param trial: The current Optuna trial
    :return: The model with the hyperparameters tuned by Optuna
    """

    # Generate layers between 2 and 6 layers according to optuna's trial
    layers_count = trial.suggest_int("Layers Count", 2, 6)
    layers = suggest_layers(trial, layers_count)

    # Optuna chooses an optimizer
    opt_choice = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    learning_rate = trial.suggest_float("learning_rate", 0.0001, 0.01)
    optimizer = get_optimizer(opt_choice, learning_rate)

    # Define the base model and the input dimension
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
        activation = trial.suggest_categorical("activation_layer_{}".format(i), ["relu", "tanh"])
        if i == 0:
            layers.append(Dense(neurons, activation=activation, input_dim=26))
        else:
            layers.append(Dense(neurons, activation=activation))
    layers.append(Dense(1, activation='sigmoid'))

    return layers


def score(model: Sequential, x, y) -> float:
    """
    Scores the model using the validation set.
    @param model: the model to score
    @param x: the validation set's features
    @param y: the validation set's target
    @return: the model's accuracy score.
    """

    score = model.evaluate(x, y)
    return score[1]


def objective(trial: Trial) -> float:
    """
    The objective function to be minimized by Optuna.
    @param trial: the current Optuna trial
    @return: float: the model's accuracy score on the validation set with the given hyperparameters and cross validation.
    """
    clear_session()
    # load the best dataset:
    x_train, y_train, x_val, y_val = load_best_dataset()

    # define the model:
    model = generate_model(trial=trial)

    # use cross validation:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = []
    for train_index, test_index in cv.split(x_train, y_train):
        # split the data:
        x_train_cv, x_test_cv = x_train.iloc[train_index], x_train.iloc[test_index]
        y_train_cv, y_test_cv = y_train.iloc[train_index], y_train.iloc[test_index]

        # fit the model:
        model.fit(x_train_cv, y_train_cv, epochs=trial.suggest_int("epochs", 10, 100),
                  batch_size=trial.suggest_int("batch_size", 8, 128), verbose=0)

        # evaluate the model:
        acc_score = model.evaluate(x_test_cv, y_test_cv, verbose=0)

        # append the score:
        cv_scores.append(acc_score[1])

    trial.report(sum(cv_scores) / len(cv_scores), 1)

    val_score = model.evaluate(x_val, y_val)

    trial.report(val_score[1], 2)

    return sum(cv_scores) / len(cv_scores)


def evaluate_best_model() -> None:
    """
    Evaluates the best model found by the tuner and saves the results.
    """
    """
    Params from 5 tries: 
    Layers Count: 3
    layer_0: 97
    activation_layer_0: tanh
    layer_1: 45
    activation_layer_1: relu
    layer_2: 112
    activation_layer_2: tanh
    optimizer: rmsprop
    learning_rate: 0.004446683137584281
    epochs: 96
    batch_size: 74
    """
    # load the best dataset:
    x_train, y_train, x_val, y_val = load_best_dataset()
    # create the model with the best parameters:
    model = Sequential()
    model = create_model_with_layers(model, layers=[Dense(97, activation='tanh', input_dim=26),
                                                    Dense(45, activation='relu'),
                                                    Dense(112, activation='tanh'),
                                                    Dense(1, activation='sigmoid')],
                                     optimizer=RMSprop(learning_rate=0.004446683137584281))
    # fit the model:
    model.fit(x_train, y_train, epochs=96, batch_size=74, verbose=0)
    # evaluate the model on the f1 score:
    y_pred = model.predict(x_val)
    y_pred = (y_pred > 0.5)
    results = evaluate_model(y_val, y_pred)
    save_evaluation_results(results, "convolutional neural network", neural_tuned_results_path,
                            "best_cnn_model_evaluation_results")

@measure_time
def main() -> None:
    """
    The main function, runs the Optuna optimization.
    @return: None. Saves the best model to a file in the neural_networks results folder.
    """
    # set the sampler:
    sampler = TPESampler(seed=42)
    # set the pruner:
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    # set the study:
    study = optuna.create_study(study_name="cnn_tuning", direction="maximize", sampler=sampler, pruner=pruner)
    # run the study:
    study.optimize(objective, n_trials=NUMBER_OF_TRIALS, n_jobs=-1, show_progress_bar=False)

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
    print(f"  Value: {trial.value} ")

    # Show the parameters of the model with the best trial
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")

    # save the results:
    study.trials_dataframe().to_csv(Path(neural_tuned_results_path, "neural_network_tuner.csv"))


if __name__ == '__main__':
    # main()
    evaluate_best_model()