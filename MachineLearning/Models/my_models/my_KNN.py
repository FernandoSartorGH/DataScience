# Imports
import numpy as np
import pandas as pd
from collections import Counter


### KNN

# Distances

## euclidian distance
def euclidian_distance(x1, x2):
    distance = np.sqrt(np.sum((x1-x2)**2))
    return distance

## manhattan distance (special case of the norm distance wich p = 1)
def manhatan_distance(x1, x2):
    #distance = np.linalg.norm(x1-x2, ord=1)
    distance = np.sum(abs(x1-x2))
    return distance

## hamming distance compute the count of differences
def hamming_distance(x1, x2): 
    # Start with a distance of zero, and count up
    distance = 0
    # Loop over the indices of the string
    L = len(x1)
    for i in range(L):
        # Add 1 to the distance if these two characters are not equal
        if x1[i] != x2[i]:
            distance += 1
    # Return the final count of differences
    return distance


## Normalize and standard data


## Accuracy
### Classifier
def accuracy(predictions, y_test):
     # calc accuracy
    accuracy = np.sum(predictions == y_test) / len(y_test)
    return accuracy

def confusionMatrix(y_test, y_pred):
    confusionMatrix = pd.crosstab(y_test, np.array(y_pred), rownames=['Actual'], colnames=['Predicted'], margins = True)
    return confusionMatrix

def ROC(self):
    pass


## Regressor

# mean squared error
def mse(y_test, predictions):
    mse = np.mean((y_test - predictions)**2)
    return mse

# r2
def r2(y_test, predictions):
    sqt = sum((y_test - np.mean(y_test))**2)
    sqr = sum((predictions - np.mean(y_test))**2)
    r2 = sqr/sqt
    return r2

# KNN Classifier
class my_KNNClassifier:

    def __init__(self, k=3, distance='euclidian'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):       
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        # choose the distance
        if self.distance == 'euclidian':
            chosen_distance = euclidian_distance
        elif self.distance == 'manhatan':
            chosen_distance = manhatan_distance
        elif self.distance == 'hamming':
            chosen_distance = hamming_distance

        # compute the distance
        distances = [chosen_distance(x, X_train) for X_train in self.X_train]

        # get the closest k (argsort returns the original index position)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voy
        most_common = Counter(k_nearest_labels).most_common()
        return most_common[0][0]



# KNN Regressor
class my_KNNRegressor:

    def __init__(self, k=3, distance='euclidian'):
        self.k = k
        self.distance = distance

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):       
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):

        # choose the distance
        if self.distance == 'euclidian':
            chosen_distance = euclidian_distance
        elif self.distance == 'manhatan':
            chosen_distance = manhatan_distance
        elif self.distance == 'hamming':
            chosen_distance = hamming_distance

        # compute the distance
        distances = [chosen_distance(x, X_train) for X_train in self.X_train]

        # get the closest k (argsort returns the original index position)
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        # majority voy
        mean_k_nearst = np.mean(k_nearest_labels)
        return mean_k_nearst

