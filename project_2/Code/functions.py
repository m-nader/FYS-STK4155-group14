import numpy as np
from sklearn.metrics import accuracy_score

# Runge function

def runge(x_values):
    """ Arguments: x_values, or a single x_value
    returns: the runge function of the x-values"""
    return 1 / (1 + 25* (x_values**2))

# Ordinary Least Squares and Ridge Regression parameter estimations

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

def OLS_gradient(X, y, theta):
    n = y.shape[0]   
    gradient = 2.0/n * (X.T @ X @ theta - X.T @ y)
    return gradient

def Ridge_gradient(X, y, theta, lam):
    n = y.shape[0]   
    gradient = 2.0/n * (X.T @ (X @ theta) - X.T @ y) + 2*lam*theta
    return gradient

# Activation functions and their derivatives

def identity(x):
    return x

def identity_der(x):
    return np.array([1]*len(x))

def ReLU(z):
    return np.where(z > 0, z, 0)

def ReLU_der(z):
    return np.where(z > 0, 1, 0)

def leaky_ReLU(z):
    return np.where(z > 0, z, 0.01 * z)

def leaky_ReLU_der(z):
    return np.where(z > 0, 1, 0.01)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    z_shift = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z_shift)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def softmax_der(z, softmax=softmax):
    s = softmax(z)
    return s * (1 - s)

# Loss functions and their derivatives

def MSE(y_model, y_data):   
    return np.mean((y_data - y_model) ** 2)

def mse_der(predict, target):
    return 2 * (predict - target) / target.shape[0]

def cross_entropy(predict, target, eps=1e-12):
    # clip to avoid log(0) and keep probs in (eps, 1)
    p = np.clip(predict, eps, 1.0)
    # pick the probability of the correct class for each sample
    n = p.shape[0]
    correct_class_probs = p[np.arange(n), target]
    return -np.mean(np.log(correct_class_probs))



def cross_entropy_der(predict, target):
    return - (target / predict) / target.shape[0]

# Performance metrics

def R2(y_model, y_data):
    return 1 - np.sum((y_data - y_model) ** 2) / np.sum((y_data - np.mean(y_model)) ** 2)

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)