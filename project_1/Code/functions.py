import numpy as np

def runge(x_values):
    """ Arguments: x_values, or a single x_value
    returns: the runge function of the x-values"""

    return 1 / (1 + 25* (x_values**2))

def OLS_parameters(X, y):
    return (np.linalg.pinv(X.T @ X) @ X.T ) @ y

def Ridge_parameters(X, y, lmbda):
    return np.linalg.inv(X.T @ X + lmbda * np.eye(X.shape[1])) @ X.T @ y

def polynomial_features(x, p):
    n = len(x)
    X = np.zeros((n, p + 1))
    X[:, 0] = 1.0
    for i in range(1,p+1):
        X[:, i] = x**i
    return X

def R2(y_data, y_model):
    # from week 34 lecture notes
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_model)) ** 2)

def MSE(y_data, y_model):   
    # from week 34 lecture notes
    return np.mean((y_data - y_model) ** 2)

def OLS_gradient_Manuela(X, y, theta):
    n = y.shape[0] 
    residuals = y - X @ theta  
    gradient = -2/n * residuals.T @ X  
    return gradient

def Ridge_gradient_Manuela(X, y, theta, lmbda):
    n = y.shape[0]  
    residuals = y - X @ theta  # Calculate residuals
    gradient = (-2/n * X.T @ residuals) + (2 * lmbda * theta)  
    return gradient

def OLS_gradient_Helene(X, y, theta):
    n = y.shape[0]   
    gradient = 2.0/n * (X.T @ X @ theta - X.T @ y)
    return gradient

def Ridge_gradient_Helene(X, y, theta, lam):
    n = y.shape[0]   
    gradient = 2.0/n * (X.T @ X @ theta - X.T @ y) + 2*lam*theta
    return gradient