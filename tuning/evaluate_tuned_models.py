# Library to build the best models with the results of the tuning:
# Data manipulation:
from pathlib import Path
import numpy as np
import pandas as pd
# Modelling:
from keras import Sequential
from keras.layers import Dense, Dropout
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score
from sklearn.svm import SVC

# Global variables:
from config import neural_networks_balanced_results_path, data_path


def build_trained_tuned_gradient_booster() -> GradientBoostingClassifier:
    pass


def build_trained_tuned_neural_network(x_train: pd.DataFrame, y_train: pd.Series) -> Sequential:
    """
    This function builds the best neural network model we found using undersampling,
    minmax scaling and dropping missing values.
    @return: the best neural network model.
    """
    model = Sequential()
    model.add(Dense(units=508, activation='relu'))
    model.add(Dropout(0.158))
    model.add(Dense(units=164, activation='relu'))
    model.add(Dropout(0.158))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dropout(0.158))
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(x_train, y_train, epochs=100, batch_size=126, verbose=0)

    return model


def build_trained_tuned_svc() -> SVC:
    pass


def test_tuned_models_on_test_set(training: pd.DataFrame, testing: pd.DataFrame) -> None:
    """
    This function tests the tuned models on the test set.
    @param x_train: the features of the training set.
    @param y_train: the target of the training set.
    @param x_test: the features of the test set.
    @param y_test: the target of the test set.
    @return: None. Prints the results of the models.
    """
    # split the data:
    x_train = training.drop('default', axis=1)
    y_train = training['default']
    x_test = testing.drop('default', axis=1)
    y_test = testing['default']

    # gradient booster classifier:
    # __________________________________________________________________________________________________

    # neural network:
    # __________________________________________________________________________________________________
    # load, and train the tuned neural network:
    model = build_trained_tuned_neural_network(x_train, y_train)

    # save the model:
    model.save(Path(neural_networks_balanced_results_path, 'neural_network_tuned.h5'))

    # evaluate the model using f1 score:
    y_pred = model.predict(x_test)
    y_pred = np.where(y_pred > 0.5, 1, 0)
    f1 = f1_score(y_test, y_pred)

    # print the results:
    print(f'The tuned neural network model has an f1 score of {round(f1, 3)} on the test set.')

    # save the results:
    with open(Path(neural_networks_balanced_results_path, 'neural_network_tuned_results.txt'), 'w') as f:
        f.write(f'The tuned neural network model has an f1 score of {round(f1, 3)} on the test set.')

    # support vector machine classifier:
    # __________________________________________________________________________________________________


def tuning_main() -> None:
    """
    This function is the main function of the script.
    @return: None. Prints the results of the models.
    """
    # load the data:
    # Note the files do not exist yet:
    training = pd.read_csv(Path(data_path, 'final_training.csv'))
    testing = pd.read_csv(Path(data_path, 'final_testing.csv'))

    # test the tuned models:
    test_tuned_models_on_test_set(training, testing)


if __name__ == '__main__':
    tuning_main()

