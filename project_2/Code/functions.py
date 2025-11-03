import numpy as np
from sklearn.metrics import accuracy_score

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

def OLS_gradient(X, y, theta):
    n = y.shape[0]   
    gradient = 2.0/n * (X.T @ X @ theta - X.T @ y)
    return gradient

def Ridge_gradient(X, y, theta, lam):
    n = y.shape[0]   
    gradient = 2.0/n * (X.T @ (X @ theta) - X.T @ y) + 2*lam*theta
    return gradient

def Lasso_gradient(X, y, theta, lam):
    n = y.shape[0]
    gradient = 2.0/n * (X.T @ (X @ theta) - X.T @ y) + lam * np.sign(theta)
    return gradient

# Defining some activation functions

def ReLU(z):
    # from week 42 exercises
    return np.where(z > 0, z, 0)


def sigmoid(z):
    # from week 42 exercises
    return 1 / (1 + np.exp(-z))

# Derivative of the ReLU function
def ReLU_der(z):
    # from week 42 exercises
    return np.where(z > 0, 1, 0)

# Derivative of sigmoid
def sigmoid_der(z):
    return sigmoid(z) * (1 - sigmoid(z))


def softmax(z):
    """Compute softmax values for each set of scores in the rows of the matrix z.
    Used with batched input data."""
    e_z = np.exp(z - np.max(z, axis=0))
    return e_z / np.sum(e_z, axis=1)[:, np.newaxis]


def softmax_vec(z):
    """Compute softmax values for each set of scores in the vector z.
    Use this function when you use the activation function on one vector at a time"""
    e_z = np.exp(z - np.max(z))
    return e_z / np.sum(e_z)

def softmax_der(z, softmax=softmax):
    s = softmax(z)
    return s * (1 - s)


def mse_der(predict, target):
    return 2 * (predict - target) / target.size



def cross_entropy(predict, target):
    return np.sum(-target * np.log(predict))

def cross_entropy_der(predict, target):
    return - (target / predict) / target.shape[0]

def cost(input, layers, activation_funcs, target):
    predict = feed_forward_batch(input, layers, activation_funcs)
    return cross_entropy(predict, target)

def accuracy(predictions, targets):
    one_hot_predictions = np.zeros(predictions.shape)

    for i, prediction in enumerate(predictions):
        one_hot_predictions[i, np.argmax(prediction)] = 1
    return accuracy_score(one_hot_predictions, targets)
# neural network functions

def create_layers(network_input_size, layer_output_sizes):
    """Creates layers batched"""
    layers = []

    i_size = network_input_size
    for layer_output_size in layer_output_sizes:
        W = np.random.randn(i_size, layer_output_size)
        b = np.random.randn(layer_output_size).T
        layers.append((W, b))

        i_size = layer_output_size
    return layers

def feed_forward_batch(inputs, layers, activation_funcs):
    """Feeds forward batched"""
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        z = np.dot(a, W) + b
        a = activation_func(z)
    return a

def feed_forward_saver(inputs, layers, activation_funcs):
    layer_inputs = []
    zs = []
    a = inputs
    for (W, b), activation_func in zip(layers, activation_funcs):
        layer_inputs.append(a)
        z = np.dot(a, W) + b
        a = activation_func(z)

        zs.append(z)

    return layer_inputs, zs, a

def backpropagation(
    inputs, layers, activation_funcs, targets, activation_ders, cost_der=mse_der
):
    layer_inputs, zs, predicts = feed_forward_saver(inputs, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        print('Backpropagation - Layer:', i)
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]

        if i == len(layers) - 1:
            # For last layer we use cost derivative as dC_da(L) can be computed directly
            print('Last layer')
            dC_da = cost_der(predicts, targets)
        else:
            # For other layers we build on previous z derivative, as dC_da(i) = dC_dz(i+1) * dz(i+1)_da(i)
            print('No last layer')
            (W, b) = layers[i + 1]
            print('W shape:', W.shape)
            dC_da = W @ dC_dz

        dC_dz = activation_der(z) * dC_da
        dC_dW = np.outer(layer_input, dC_dz)
        dC_db = dC_dz
        print('dC_dW shape:', dC_dW.shape)
        print('dC_db shape:', dC_db.shape)
        print('dC_dz shape:', dC_dz.shape)
        print('dC_da shape:', dC_da.shape)

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads

def backpropagation_cross(
    inputs, layers, activation_funcs, targets, activation_ders
):
    # backpropagation with simplification for softmax + cross-entropy
    layer_inputs, zs, predicts = feed_forward_saver(inputs, layers, activation_funcs)

    layer_grads = [() for layer in layers]

    # We loop over the layers, from the last to the first
    for i in reversed(range(len(layers))):
        layer_input, z, activation_der = layer_inputs[i], zs[i], activation_ders[i]
        if i == len(layers) - 1:
            dC_dz = predicts - targets  # Simplification for softmax + cross-entropy
        else:
            (W, b) = layers[i + 1]
            dC_da = W @ dC_dz.T
            dC_dz = activation_der(z) * dC_da.T 
        dC_dW = layer_input.T @ dC_dz
        dC_db = np.mean(dC_dz, axis=0)

        layer_grads[i] = (dC_dW, dC_db)

    return layer_grads


def train_network(
    inputs, layers, activation_funcs, activation_ders, targets, cost_der, learning_rate=0.001, epochs=100, momentum=0.3, minibatch_size=15
):
    change = [ (np.zeros_like(W), np.zeros_like(b)) for (W, b) in layers ]
    n_data = inputs.shape[0]
    m = int(n_data / minibatch_size)
    for i in range(epochs):
        indices = np.random.permutation(n_data)
        x_shuffled = inputs[indices]
        y_shuffled = targets[indices]
        for i in range(m):
            xi = x_shuffled[i : i + minibatch_size]
            yi = y_shuffled[i : i + minibatch_size]
            layer_grads = backpropagation(xi, layers, activation_funcs, yi, activation_ders, cost_der)
            for idx, ((W, b), (W_g, b_g), (W_c, b_c)) in enumerate(zip(layers, layer_grads, change)):
                new_change_W = learning_rate * W_g + momentum * W_c
                new_change_b = learning_rate * b_g + momentum * b_c
                W -= new_change_W[idx]
                b -= new_change_b[idx]
                layers[idx] = (W, b)
                change[idx] = (new_change_W, new_change_b)  
