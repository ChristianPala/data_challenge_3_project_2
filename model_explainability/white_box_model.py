# Library to explain a tuned decison tree, as our first model explanation:
# Libraries
from sklearn.tree import DecisionTreeClassifier
from feature_selection.remove_correlated_features import simplify_dataset

# plotting:
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt


# Global variables:
from config import final_train_under_csv_path, final_val_under_csv_path, white_box_model_explanation_path

# Ensure the directory exists:
white_box_model_explanation_path.mkdir(parents=True, exist_ok=True)


def white_box_main() -> None:
    """
    Function to explain the white box models.
    :return: None. Saves the explanations.
    """

    # Create the tuned decision tree:
    model = DecisionTreeClassifier(criterion='gini', max_depth=9, min_samples_leaf=1, min_samples_split=2)

    # get the simplified dataset:
    x_train, y_train, x_val, y_val = simplify_dataset(final_train_under_csv_path, final_val_under_csv_path)

    # Fit the model:
    model.fit(x_train, y_train)

    # plot the decision tree:
    plt.figure(figsize=(25, 20))
    plot_tree(model, feature_names=x_train.columns, class_names=['0', '1'], filled=True,
              rounded=True, fontsize=8, proportion=True, max_depth=3)
    plt.title('Decision Tree white box model')
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


if __name__ == '__main__':
    white_box_main()