# Data manipulation:
import pandas as pd

# Tuning
import optuna
from optuna.trial import TrialState, Trial
from tuning.balanced_neural_network import fit_model, predict_model, evaluate_model
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

# Modelling:
from keras.models import Model, Sequential
from keras.layers import Dense, Layer, Dropout
from keras.backend import clear_session
from keras.optimizers import Adam, RMSprop, SGD, Optimizer
from pathlib import Path
from modelling.train_test_validation_split import split_data
from sklearn.model_selection import StratifiedKFold

# Timing:
from auxiliary.method_timer import measure_time

# Global variables:
from config import balanced_datasets_path, neural_tuned_results_path
from config import scaled_datasets_path, undersampled_datasets_path


def load_best_dataset(path: Path = Path(balanced_datasets_path, "smote_enn", "robust_scaler_scaling_drop",)) \
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


def create_model_with_layers(model: Model, layers: list[Layer], dropout: float = 0.1,
                             optimizer: Optimizer = Adam(), loss: str = 'binary_crossentropy',
                             metrics: list[str] = ['accuracy']) -> Model:
    if metrics is None:
        metrics = ['accuracy']
    compiled_model = model
    for i in range(len(layers)):
        compiled_model.add(layers[i])
        if i < len(layers)-1:
            compiled_model.add(Dropout(dropout))

    compiled_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

    return compiled_model


def get_optimizer(choice: str, learning_rate: float = 0.001) -> Optimizer:
    if choice == "adam":
        return Adam(learning_rate=learning_rate)
    elif choice == "rmsprop":
        return RMSprop(learning_rate=learning_rate)
    return SGD(learning_rate=learning_rate)


def generate_model(trial: Trial) -> Model:
    """
    Generates a model with a number of levels and an optimizer chosen by Optuna.
    :return: The model with the hyperparameters tuned by Optuna
    """
    
    # Generate layers between 1 and 6 layers according to Optuna's trial
    layers_count = trial.suggest_int("Layers Count", 1, 6)
    layers = suggest_layers(trial, layers_count)

    # Optuna chooses an optimizer
    opt_choice = trial.suggest_categorical("optimizer", ["adam", "rmsprop", "sgd"])
    learning_rate = trial.suggest_float("learning_rate", 0.00001, 0.1)
    optimizer = get_optimizer(opt_choice, learning_rate)

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
        neurons = trial.suggest_int("layer_{}".format(i), 10, 300)
        activation = trial.suggest_categorical("activation_layer_{}".format(i), ["relu", "tanh"])
        if i == 0:
            layers.append(Dense(neurons, activation=activation, input_dim=23))
        else:
            layers.append(Dense(neurons, activation=activation))
    layers.append(Dense(1, activation='sigmoid'))
        
    return layers


def score(model: Sequential, x, y) -> float:

    score = model.evaluate(x, y)
    return score[1]


def objective(trial: Trial):
    clear_session()
    # load the best dataset:
    #x_train, y_train, x_val, y_val = load_best_dataset()

    csv_files: list[Path] = list(scaled_datasets_path.glob('*.csv'))
    df = pd.read_csv(csv_files[0])
    test_path = Path(undersampled_datasets_path, "minmax_scaler_scaling_drop", "test.csv")
    test = pd.read_csv(test_path)

    # split the data into train and test:
    x_train, x_val, _, y_train, y_val, _ = split_data(df, 'default', validation=True)

    x_test, x_val, _, y_test, y_val, _ = split_data(test, 'default', validation=True)

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
        score = model.evaluate(x_test_cv, y_test_cv, verbose=0)

        # append the score:
        cv_scores.append(score[1])
        
    trial.report(sum(cv_scores) / len(cv_scores), 1)

    val_score = model.evaluate(x_val, y_val)
    
    trial.report(val_score[1], 2)
    y_pred = predict_model(model, x_test)
    test_score = evaluate_model(y_test, y_pred)

    trial.report(test_score[0], 3)

    # return the score on the validation dataset:
    return test_score[0]


@measure_time
def main():
    # set the sampler:
    sampler = TPESampler(seed=42)
    # set the pruner:
    pruner = MedianPruner(n_startup_trials=10, n_warmup_steps=5)
    # set the study:
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    # run the study:
    study.optimize(objective, n_trials=10, n_jobs=-1, show_progress_bar=False)

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
    study.trials_dataframe().to_csv(Path(neural_tuned_results_path, "neural_network_tuner.csv"))


if __name__ == '__main__':
    main()