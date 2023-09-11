import numpy as np

def LinearRegression(X, Y, lr, n_iters):
    n_samples = X.shape[0]  
    weights = np.zeros(1) 
    print(weights)
    bias = 0

    for _ in range(n_iters):
        y_hat = np.dot(X, weights) + bias

        dw = (1/n_samples)*(np.dot(X.T, (y_hat - Y)))
        db = (1/n_samples)*(np.sum(y_hat - Y))

        weights = weights - lr*dw
        bias = bias - lr*db
    
    return y_hat, weights, bias

def predict(X, weights, bias):
    y_predicted = np.dot(X, weights) + bias
    return y_predicted

X = np.array([[1], [2], [3], [4], [5]])
Y = np.array([1, 2, 3, 4, 5])

y_hat, weights, bias = LinearRegression(X, Y, 0.01, 1000)

predictions = predict(np.array([10]), weights, bias)
print(predictions)
