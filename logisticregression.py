import numpy as np

def LogisticRegression(X, Y, n_iters, lr):
    
    n_samples = X.shape[0]
    weights = np.zeros(X.shape[1])
    bias = np.random.rand()
    
    for _ in range(n_iters):

        linear_term = np.dot(X, weights) + bias
        y_hat = 1/(1+np.exp(-linear_term))

        dw = (1/n_samples) * np.dot(X.T, (y_hat - Y))
        db = (1/n_samples) * np.sum(y_hat - Y)

        weights = weights - lr*dw
        bias = bias - lr*db

    return weights, bias

def predict(X, weights, bias):
    linear_term = np.dot(X, weights) + bias
    y_hat = 1/(1+np.exp(-linear_term))
    return y_hat

X = np.array([[1], [2], [3], [4]])
Y = np.array([0, 1, 0, 1])
weights, bias = LogisticRegression(X, Y, 1000, 0.01)
print(predict(np.array([3]), weights, bias))