# Auxiliary library to create and train the final models from the tuning results:
# Libraries:
# Data manipulation:
from pathlib import Path
import pandas as pd
import numpy as np
# Plotting:
import matplotlib.pyplot as plt
# Modelling:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from keras import Model, Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense
from keras.optimizers import Adam

# Global variables:
from config import final_train_under_csv_path, final_val_under_csv_path, final_models_path, final_test_under_csv_path
from tuning.convolutional_neural_network_tuner_Optuna import create_model_with_layers

# Ensure the directory exists:
final_models_path.mkdir(parents=True, exist_ok=True)


def gradient_boosting_model(x_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """
    This function creates a gradient boosting model.
    :return: GradientBoostingClassifier: the model.
    Best parames from the tuner:

    {'ccp_alpha': 0.0, 'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.2,
    'loss': 'log_loss', 'max_depth': 10, 'max_features': None, 'max_leaf_nodes': None,
    'min_impurity_decrease': 0.0, 'min_samples_leaf': 5, 'min_samples_split': 5,
    'min_weight_fraction_leaf': 0.0, 'n_estimators': 800, 'n_iter_no_change': None, '
    random_state': 42, 'subsample': 1.0, 'tol': 0.0001, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}
    """
    gb = GradientBoostingClassifier(ccp_alpha=0.0, criterion='friedman_mse',
                                    init=None, learning_rate=0.2, loss='log_loss',
                                    max_depth=10, max_features=None, max_leaf_nodes=None,
                                    min_impurity_decrease=0.0, min_samples_leaf=5,
                                    min_samples_split=5, min_weight_fraction_leaf=0.0,
                                    n_estimators=800, n_iter_no_change=None, random_state=42,
                                    subsample=1.0, tol=0.0001, validation_fraction=0.1, verbose=0,
                                    warm_start=False)

    gb.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(gb, Path(final_models_path, "gradient_boosting_model.pkl"))
    return gb


def neural_network_model(x_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    This function creates a neural network model.
    :return: Model: the model.
    """
    """
    Best trial:
    Value: 0.8298506259918212 
      Params: 
        Layers Count: 2
        layer_0: 23
        activation_layer_0: relu
        filters_layer_0: 140
        kernel_size_layer_0: 7
        pool_size_layer_0: 3
        layer_1: 133
        activation_layer_1: tanh
        optimizer: adam
        learning_rate: 0.006679385448500152
        epochs: 73
        batch_size: 93
    """
    # create the tuned model:
    model = Sequential()
    model = create_model_with_layers(model, layers=[
                                            Conv1D(filters=140, kernel_size=7, activation='tanh', input_shape=(26, 1)),
                                            MaxPooling1D(pool_size=3),
                                            Flatten(),
                                            Dense(133, activation='tanh'),
                                            Dense(1, activation='sigmoid')
                                        ], optimizer=Adam(learning_rate=0.006679385448500152))
    # fit the model:
    model.fit(x_train, y_train, epochs=73, batch_size=93, verbose=0)

    # save the model as a .h5 file and as a pickle file:
    model.save(Path(final_models_path, "cnn_model.h5"))
    pd.to_pickle(model, Path(final_models_path, "cnn_model.pkl"))

    return model


def supper_vector_machine_model(x_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    This function creates a support vector machine model.
    :return: SVC: the model.
    """
    svc = SVC(random_state=42, C=2.0, kernel='rbf', probability=True)
    svc.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(svc, Path(final_models_path, "support_vector_machine_model.pkl"))
    return svc


def create_final_models_main() -> None:
    """
    This function is the main function.
    :return: None
    """
    # load the data:
    training = pd.read_csv(final_train_under_csv_path)
    validation = pd.read_csv(final_val_under_csv_path)
    testing = pd.read_csv(final_test_under_csv_path)

    # concatenate training and validation:
    training = pd.concat([training, validation], axis=0)

    # create the final models:
    x_train = training.drop("default", axis=1)
    y_train = training["default"]

    # create and save the models:
    gb = gradient_boosting_model(x_train, y_train)
    cnn = neural_network_model(x_train, y_train)
    svc = supper_vector_machine_model(x_train, y_train)

    # evaluate the models on the test set:
    x_test = testing.drop("default", axis=1)
    y_test = testing["default"]
    print("Gradient Boosting Model:")
    gb_pred = gb.predict(x_test)
    print("F1 score: ", f1_score(y_test, gb_pred))
    class_rep_gb = classification_report(y_test, gb_pred)
    cm_gb = ConfusionMatrixDisplay(confusion_matrix(y_test, gb.predict(x_test)))
    # save the confusion matrix:
    cm_gb.plot()
    cm_gb.figure_.savefig(Path(final_models_path, "gb_confusion_matrix.png"))
    plt.close(cm_gb.figure_)

    cnn_predictions = cnn.predict(x_test)
    cnn_predictions = np.where(cnn_predictions > 0.5, 1, 0)
    class_rep_cnn = classification_report(y_test, cnn_predictions)
    print("F1 score: ", f1_score(y_test, cnn_predictions))
    cm_nn = ConfusionMatrixDisplay(confusion_matrix(y_test, cnn_predictions))
    # save the confusion matrix:
    cm_nn.plot()
    cm_nn.figure_.savefig(Path(final_models_path, "nn_confusion_matrix.png"))

    print("Support Vector Machine Model:")
    svc_predictions = svc.predict(x_test)
    svc_predictions = np.where(svc_predictions > 0.5, 1, 0)
    class_rep_svc = classification_report(y_test, svc_predictions)
    print("F1 score: ", f1_score(y_test, svc_predictions))
    cm_svc = ConfusionMatrixDisplay(confusion_matrix(y_test, svc_predictions))
    # save the confusion matrix:
    cm_svc.plot()
    cm_svc.figure_.savefig(Path(final_models_path, "svc_confusion_matrix.png"))

    # save the classification reports:
    with open(Path(final_models_path, "classification_report.txt"), "w") as f:
        f.write("Gradient Boosting Model:\n")
        f.write(class_rep_gb)
        f.write("\n\n")
        f.write("Neural Network Model:\n")
        f.write(class_rep_cnn)
        f.write("\n\n")
        f.write("Support Vector Machine Model:\n")
        f.write(class_rep_svc)
        f.write("\n\n")


if __name__ == '__main__':
    create_final_models_main()
