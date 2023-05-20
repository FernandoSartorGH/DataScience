# References
 # Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow - Aurélien Géron
 # Machine learning - Fast reference guide - Matt Harrison
 # https://www.youtube.com/@patloeber
 # https://www.youtube.com/@Dataquestio


# STEPS

# Subtract the mean from X
# Calculate Cov(X,X)
# Calculate eigenvectors and eigenvalues of the covariance matrix
# Sort the eigen vectors acording the their eigen values 
# Transform the original n-dimensional data into k dimension


import numpy as np

class PCA:

    def __init__(self, n_components):
        self.n_components = n_components
        self.eigenvectors = None
        
    def fit(self, X):
        
        # mean centering
        X = (X -  X.mean())/X.std()

        # covariance, functions needs samples as columns
        cov = np.cov(X.T, bias=1)

        # eigenvectors, eigenvalues
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        # sort by eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors

    def transform(self, X):
        
        # mean centering
        X = (X -  X.mean())/X.std()

        pca = np.dot(np.array(X), self.components)
        pca  = pca[:,:self.n_components]
        
        return pca
