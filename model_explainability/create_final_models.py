# Auxiliary library to create and train the final models from the tuning results:
# Libraries:
# Data manipulation:
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from keras import Model
# Modelling:
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, confusion_matrix, ConfusionMatrixDisplay
from modelling.neural_network import fit_model, create_convolutional_model
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Global variables:
from config import final_train_csv_path, final_val_csv_path, final_models_path, final_test_csv_path

final_models_path.mkdir(parents=True, exist_ok=True)


def gradient_boosting_model(x_train: np.ndarray, y_train: np.ndarray) -> GradientBoostingClassifier:
    """
    This function creates a gradient boosting model.
    :return: GradientBoostingClassifier: the model.
    """
    gb = GradientBoostingClassifier(learning_rate=0.22271924404121154, loss='exponential', max_depth=7,
                                    max_leaf_nodes=4, min_impurity_decrease=0.35176743416487977,
                                    min_samples_leaf=1, min_samples_split=5,
                                    min_weight_fraction_leaf=0.000723912587745684, n_estimators=593,
                                    subsample=0.7424194687719635, random_state=42)

    gb.fit(x_train, y_train)
    # save the model:
    pd.to_pickle(gb, Path(final_models_path, "gradient_boosting_model.pkl"))
    return gb


def neural_network_model(x_train: np.ndarray, y_train: np.ndarray) -> Model:
    """
    This function creates a neural network model.
    :return: Model: the model.
    """
    # create the model:
    model = create_convolutional_model(x_train.shape[1])  # Todo add all parameters from the tuning results
    # fit the model:
    model = fit_model(model, x_train, y_train)
    # save the pickle of the model:
    pd.to_pickle(model, Path(final_models_path, "cnn_model.pkl"))
    return model


def supper_vector_machine_model(x_train: np.ndarray, y_train: np.ndarray) -> SVC:
    """
    This function creates a support vector machine model.
    :return: SVC: the model.
    """
    svc = SVC(random_state=42, C=4900, gamma=10 ** -5, probability=True)
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
    training = pd.read_csv(final_train_csv_path)
    validation = pd.read_csv(final_val_csv_path)
    testing = pd.read_csv(final_test_csv_path)

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
