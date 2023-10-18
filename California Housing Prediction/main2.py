import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


X , y = fetch_california_housing(return_X_y=True)
print()
# Adding a vector of ones to the data matrix to absorb the bias term -> Shape (20640, 9) with first column being 1's
X = np.hstack([np.ones([X.shape[0], 1]), X])

# From now on, D refers to the number of features in the augmented dataset (i.e. including the dummy '1' feature for the absorbed bias term)

# Split into train and test
test_size = 0.9 # we select a relatively large test set due to the large size of the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=1)


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
    N = len(y)
    D = len(X[0, :])
    print(D)
    Phi = np.zeros([N, D])
    for n in range(N):
        for d in range(D):
            if d <= 7:
                #Phi[n, d] = X[n, d]**(d/D)
                Phi[n, d] = np.log(X[n, d])
            else:
                Phi[n, d] = X[n, d]

    return np.linalg.pinv(Phi)@y


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
    
    Reference: `https://en.wikipedia.org/wiki/Mean_squared_error`
    
    Parameters
    ----------
    y_true : array
        True regression targets.
    y_pred : array
        Predicted regression targets.
        
    Returns
    -------
    mse : float
        Mean squared error.
        
    """
    dimY = len(y_true)
    mse = 1/dimY * ((y_true - y_pred)**2).sum()
    return mse


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

# generally it is said that a MSE between 0.25 and 0.5 is considered good, however, truly
# the only way to judge if the value of the metric is reasonable is by comparing it to some benchmark, like the results of another model.

"""np.set_printoptions(threshold=np.inf, suppress=True)
yy = np.vstack([y_pred_ls, y_test])
print(yy.T)
"""
np.set_printoptions(threshold=np.inf, suppress=True)
print(w_ls)
print(X_test[0, :])
plt.scatter(y_test, y_pred_ls, s=1)
plt.show()