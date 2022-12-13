# Library to explain a tuned decision tree, as our first model explanation, see:
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.plot_tree.html
# Libraries
# Data manipulation:
import pandas as pd

# Modelling:
from sklearn.tree import DecisionTreeClassifier
from feature_selection.remove_correlated_features import simplify_dataset
from feature_engineering.create_features import pay_status_cumulative, total_bill_amount, total_paid_amount
from modelling.train_test_validation_split import split_data
from sklearn.metrics import precision_score, recall_score, f1_score
# Plotting:
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
# Tuning:
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# Global variables:
from config import eda_csv_file_path, white_box_model_explanation_path

# Ensure the directory exists:
white_box_model_explanation_path.mkdir(parents=True, exist_ok=True)


def white_box_main() -> None:
    """
    Function to explain the white box models.
    :return: None. Saves the explanations.
    """

    # Create the tuned decision tree:
    model = DecisionTreeClassifier()

    # get the initial dataset without scaling:
    df = pd.read_csv(eda_csv_file_path)

    # Add the feature engineering features:
    df = pay_status_cumulative(df)
    df = total_bill_amount(df)
    df = total_paid_amount(df)

    # Merge the dataframes and save the result:
    x_train, x_test, y_train, y_test = split_data(df, 'default')
    training = pd.concat([x_train, y_train], axis=1)
    testing = pd.concat([x_test, y_test], axis=1)
    training.to_csv(white_box_model_explanation_path / 'training.csv', index=False)
    testing.to_csv(white_box_model_explanation_path / 'testing.csv', index=False)

    # Simplify the dataset:
    x_train, y_train, x_test, y_test = simplify_dataset(white_box_model_explanation_path / 'training.csv',
                                                        white_box_model_explanation_path / 'testing.csv')

    # tune the model with grid search:
    param_grid = {'max_depth': [2, 3, 4, 5, 6, 7, 8, 9, 10],
                  'min_samples_leaf': [1, 2, 3, 4, 5],
                  'min_samples_split': [2, 3, 4, 5],
                  'criterion': ['gini', 'entropy']}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)

    # get the best model:
    params = grid_search.fit(x_train, y_train).best_params_
    print(params)

    model = DecisionTreeClassifier(**params)
    model.fit(x_train, y_train)

    # plot the decision tree:
    plt.figure(figsize=(25, 20))
    plot_tree(model, feature_names=x_train.columns, class_names=['0', '1'], filled=True,
              rounded=True, fontsize=12, proportion=True, max_depth=3)
    plt.title('Decision Tree white box model')
    # add legend for the colours:
    plt.legend(['0', '1'])
    plt.savefig(white_box_model_explanation_path / 'decision_tree.png')
    plt.close()

    # plot the feature importance sorted:
    feature_importance = model.feature_importances_
    feature_importance_sorted = sorted(zip(feature_importance, x_train.columns), reverse=True)
    plt.figure(figsize=(10, 10))
    plt.barh([x[1] for x in feature_importance_sorted], [x[0] for x in feature_importance_sorted])
    plt.title('Feature importance for the decision tree white box model')
    plt.xlabel('Feature importance')
    plt.ylabel('Feature')
    # give more space to the y-axis:
    plt.subplots_adjust(left=0.3)
    plt.savefig(white_box_model_explanation_path / 'feature_importance.png')

    # evaluate the model on precision, recall and f1:
    y_pred = grid_search.predict(x_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f'Precision: {precision:.3f}')
    print(f'Recall: {recall:.3f}')
    print(f'F1: {f1:.3f}')


if __name__ == '__main__':
    white_box_main()
