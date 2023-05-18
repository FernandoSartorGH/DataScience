# References
 # Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow - Aurélien Géron
 # Machine learning - Fast reference guide - Matt Harrison
 # https://www.youtube.com/@patloeber
 # https://www.youtube.com/@Dataquestio


# STEPS

# Training
## Initialize weight as zeros
## Initialize bias as zero

# Given a data point:
## Predict result by using linear equation
## Calculate error
## Use gradient descent to figure out new weight and bias values
## Repeat n times

# Testing
# Givena data point:
## Put in the values from the data point into the equation

import numpy as np
import pandas as pd



### Classifier
def accuracy(predictions, y_test):
	# calc accuracy
	acc = np.sum(predictions == y_test) / len(y_test)
	return acc

def confusionMatrix(y_test, y_pred):
    cm = pd.crosstab(y_test, np.array(y_pred), rownames=['Actual'], colnames=['Predicted'], margins = True)
    return cm

def ROC(self):
    pass


## Regressor
def mse(y_test, predictions):
    mse = np.mean((y_test - predictions)**2)
    return mse

def r2(y_test, predictions):
    sqt = sum((y_test - np.mean(y_test))**2)
    sqr = sum((y_test - predictions)**2)
    r2 = 1 - (sqr/sqt)
    return r2


# Linear Regression ------->>> Add option to estimates B0 (intercept)
class my_LinearRegression:

	def __init__(self, lr = 0.001, n_iters = 1000, fit_mode = 'ols'):
		self.lr = lr
		self.n_iters = n_iters
		self.fit_mode = fit_mode
		self.weights = None
		self.bias = None


	def fit(self, X, y):

		# set shape
		self.n_samples, self.n_features = X.shape

		# Gradiente descent
		if self.fit_mode == 'gd':

			# initialize matrix with zeros
			self.weights = np.zeros(self.n_features)
			self.bias = 0

			for _ in range(self.n_iters):

				# predict y with matrix products plus bias
				y_pred = np.dot(X, self.weights) + self.bias

				# calculate gradiente with matrix products
				dw = (1/self.n_samples) * np.dot(X.T, (y_pred-y))
				db = (1/self.n_samples) * np.sum(y_pred-y)

				# update weights and bias
				self.weights = self.weights - self.lr * dw
				self.bias = self.bias - self.lr * db


		# Ordinary Least Squares with pseudo inverse
		elif self.fit_mode == 'ols':
			
			# Add constant to matriz X
			X_b = np.c_[np.ones((X.shape[0], 1)), X]
			
			# Calculate the vector with the parameters
			self.params = np.linalg.pinv(X_b).dot(y)


	def predict(self, X):

		if self.fit_mode == 'gd':
			y_pred = np.dot(X, self.weights) + self.bias
			return y_pred

		else:
			X_b = np.c_[np.ones((X.shape[0],1)), X]
			y_pred = X_b.dot(self.params)
			return y_pred



# Logistc Regression

# sigmoid fuction
def sigmoid(x):
    return 1/(1+np.exp(-x))

class my_LogisticRegression:

	def __init__(self, lr = 0.01, n_iters = 1000):
		self.lr = lr
		self.n_iters = n_iters
		self.weights = None
		self.bias = None


	def fit(self, X, y):

		# initialize matrix with zeros
		n_samples, n_features = X.shape
		self.weights = np.zeros(n_features)
		self.bias = 0

		for _ in range(self.n_iters):

			# predict y with matrix products plus bias
			linear_pred = np.dot(X, self.weights) + self.bias
			predictions = sigmoid(linear_pred)

			# update weights
			## w = w - a.dw
			## b = b - a.db

			# calculate gradiente with matrix products
			dw = (1/n_samples) * np.dot(X.T, (predictions - y))
			db = (1/n_samples) * np.sum(predictions - y)

			# update weights and bias
			self.weights = self.weights - self.lr * dw
			self.bias = self.bias - self.lr * db


	def predict(self, X):

		linear_pred = np.dot(X, self.weights) + self.bias
		y_pred = sigmoid(linear_pred)
		class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
		return class_pred