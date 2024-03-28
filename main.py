import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

TEST_SIZE = 0.3
K = 3


class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def predict(self, features):
        """
        Given a list of features vectors of testing examples
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors
        """
        classifier = KNeighborsClassifier(n_neighbors=K)
        classifier.fit(self.trainingFeatures, self.trainingLabels)
        return classifier.predict(features)


def load_data(filename):
    """
    Load spam data from a CSV file `filename` and convert into a list of
    features vectors and a list of target labels. Return a tuple (features, labels).

    features vectors should be a list of lists, where each list contains the
    57 features vectors

    labels should be the corresponding list of labels, where each label
    is 1 if spam, and 0 otherwise.
    """
    # the path of the file
    path = 'C:/Users/HP/Desktop/' + filename + '.csv'
    df = pd.read_csv(path)
    data = df.where((pd.notnull(df)), '')
    # split the data into features and labels
    features = data.iloc[:, :-1].values
    labels = data.iloc[:, -1].values
    return features, labels


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """
    mean = np.mean(features, axis=0)
    standard_div = np.std(features, axis=0)
    update_features = (features - mean) / standard_div
    return update_features


def train_mlp_model(features, results):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    # the first layer 10 neurons and the second layer 5
    model = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic')
    model.fit(features, results)
    return model


def evaluate(labels, predictions):
    """
    Given a list of actual labels and a list of predicted labels,
    return (accuracy, precision, recall, f1).

    Assume each label is either a 1 (positive) or 0 (negative).
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    # to print the confusion matrix
    cmatrix = confusion_matrix(labels, predictions)
    return accuracy, precision, recall, f1, cmatrix


def main():
    print("Done By: Pierre Backleh 1201296 & Mohammad Salem 1200651")
    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data('spambase')
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train the k-NN model and make predictions for it
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test)
    accuracy1, precision1, recall1, f1_1, cmatrix1 = evaluate(y_test, predictions)
    print("Confusion Matrix for k-NN:")
    print(cmatrix1)

    # Print results
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy1)
    print("Precision: ", precision1)
    print("Recall: ", recall1)
    print("F1: ", f1_1)

    # Train an MLP model and make predictions
    trained_model = train_mlp_model(X_train, y_train)
    predictions = trained_model.predict(X_test)
    accuracy, precision, recall, f1, cmatrix = evaluate(y_test, predictions)
    print("Confusion Matrix for MLP:")
    print(cmatrix)

    # Print results
    print("**** 2-MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)


if __name__ == "__main__":
    main()
