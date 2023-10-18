import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def fit_least_squares(X, y):
    """Fit ordinary least squares model to the data.
    (X^T*X)^-1 * X^T * y
    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    y : array, shape [N]
        Regression targets.
        
    Returns
    -------
    w : array, shape [D]
        Optimal regression coefficients (w[0] is the bias term).
        
    """
    return np.linalg.pinv(X)@y


def fit_ridge(X, y, reg_strength):
    """Fit ridge regression model to the data.
    (X^T*X + reg_strength*I)^-1 * X^T * y
    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    y : array, shape [N]
        Regression targets.
    reg_strength : float
        L2 regularization strength
        
    Returns
    -------
    w : array, shape [D]
        Optimal regression coefficients (w[0] is the bias term).
    
    """
    D = X.shape[1]
    return np.linalg.inv(X.T@X + reg_strength*np.eye(D))@X.T@y


def predict_linear_model(X, w):
    """Generate predictions for the given samples.
    
    Parameters
    ----------
    X : array, shape [N, D]
        (Augmented) feature matrix.
    w : array, shape [D]
        Regression coefficients.
        
    Returns
    -------
    y_pred : array, shape [N]
        Predicted regression targets for the input data.
        
    """
    return X@w


def mean_squared_error(y_true, y_pred):
    """Compute mean squared error between true and predicted regression targets.
    
    Parameters
    ----------
    y_true : array
        True regression targets.
    y_pred : array
        Predicted regression targets.
        
    Returns
    -------
    MSE : float
        Mean squared error.
        
    """
    dimY = len(y_true)
    MSE = 1/dimY * ((y_true - y_pred)**2).sum()
    return MSE

###########################################################################################################################################
# Data Analysis
X , y = fetch_california_housing(return_X_y=True)
# I will add a vector of ones to the data matrix to absorb the bias term -> Shape (20640, 9) with first column being 1's
X = np.hstack([np.ones([X.shape[0], 1]), X])
# D now refers to the number of features in the augmented dataset (i.e. including the dummy '1' feature for the absorbed bias term)
# X[:, 0] = bias terms
# X[:, 5] = block group population
# X[:, 7] = lattitude
# X[:, 8] = longitude

# Visualize the target data geographically
plt.figure(figsize=(9, 7))
plt.scatter(X[:, 7], X[:, 8], alpha=0.2, c=y, s=X[:, 5]/150)
xlabel = 'Lattitude' + chr(176)
ylabel = 'Longitude' + chr(176)
plt.xlabel(xlabel)
plt.ylabel(ylabel)
plt.colorbar(label='Median House Value (in $100,000)')
plt.show()

np.set_printoptions(threshold=np.inf, suppress=True)
#yy = np.vstack([y_pred_ls, y_test])
#print(yy.T)


###########################################################################################################################################
# Data Regression

# Split into train and test
test_size = 0.9 # we select a relatively large test set due to the large size of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)

print(X_train[2, :])
print(y_train[2])

scaler = StandardScaler().fit(X_train)

def preprocessor(X):
    A = np.copy(X)
    A = scaler.transform(A)
    return A

X_train = preprocessor(X_train)

print(X_train[2, :])
print(y_train[2])

# Ordinary least squares regression
w_ls = fit_least_squares(X_train, y_train)
y_pred_ls = predict_linear_model(X_test, w_ls)
mse_ls = mean_squared_error(y_test, y_pred_ls)
print('MSE for Least squares = {0}'.format(mse_ls))

# Ridge regression
reg_strength = 1e-2
w_ridge = fit_ridge(X_train, y_train, reg_strength)
y_pred_ridge = predict_linear_model(X_test, w_ridge)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
print('MSE for Ridge regression = {0}'.format(mse_ridge))

# Plot True vs. Predicted Target Values
plt.scatter(y_test, y_pred_ls, s=1)
plt.xlabel('True Average House Value (in $100,000)')
plt.ylabel('Predicted Average House Value (in $100,000)')
plt.show()