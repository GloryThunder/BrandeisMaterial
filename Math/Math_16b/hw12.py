# MATH 16B Spring 2023
# Name:

###############################################################
# Do NOT modify the file name or any of the function headers! #
###############################################################

import numpy as np
import pandas as pd

def predict_from_model(training, target, testing):
    """
    Implement the function predict_from_model.
    
    >>> training = testing = np.array([[1,0],
                                       [2,1],
                                       [3,1],
                                       [4,0]])
    >>> target = np.array([2,4,4,5])
    >>> predict_from_model(training, target, testing)
    array([2.15, 3.55, 4.45, 4.85])
    """
    "*** YOUR CODE HERE ***"
    A = np.column_stack((training, np.ones((training.shape[0], 1))))
    B = np.column_stack((testing, np.ones((testing.shape[0], 1))))
    x, _, _, _ = np.linalg.lstsq(A, target, rcond=None)
    predictions = np.dot(B, x)
    return predictions

#print(predict_from_model(np.array([[1,0],[2,1],[3,1],[4,0]]), np.array([2,4,4,5]), np.array([[1,0],[2,1],[3,1],[4,0]])))

def moore(file_path):
    """
    Implement the function moore.

    file_path: Can be assumed to be 'transistors.csv'.
    """
    "*** YOUR CODE HERE ***"
    data = pd.read_csv(file_path)
    log_transistors = np.log2(data["Transistors"])
    t = data["Year"] - data["Year"].min()
    ones = np.ones(t.shape, dtype=t.dtype)
    X = np.array((t, ones)).T
    gamma, beta = np.linalg.lstsq(X, log_transistors, rcond=None)[0]
    return 1.0 / gamma

#print(moore('C:/Users/Coco M1lk/Desktop/Source Code/VSCode/Python/Math 16b/transistors.csv'))

def min_validation_error(features, b):
    """
    Implement the function min_validation_error:.

    >>> features = np.array([[1, 0.01, 1],
                             [1, 1, 2],
                             [2, 1, 3.01],
                             [1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1]])
    >>> b = np.array([2, 2.01, 3, 2, 1, 1])
    >>> min_validation_error(features, b)
    1
    """
    "*** YOUR CODE HERE ***"
def min_validation_error(features, b):
    n, m = features.shape
    k_vals = range(m + 1)
    split = n // 2
    f_train = features[:split, :]
    b_train = b[:split]
    f_valid = features[split:, :]
    b_valid = b[split:]
    min_error = np.inf
    best_k = 0
    for k in k_vals:
        model = np.linalg.lstsq(np.column_stack([f_train[:, :k], np.ones((split, 1))]), b_train, rcond=None)[0]
        y_val_pred = np.column_stack([f_valid[:, :k], np.ones((n - split, 1))]) @ model
        error = np.sum((b_valid - y_val_pred) ** 2)
        if error < min_error:
            min_error = error
            best_k = k

    return best_k

#print(min_validation_error(np.array([[1, 0.01, 1],[1, 1, 2],[2, 1, 3.01],[1, 0, 0],[0, 1, 0],[0, 0, 1]]), np.array([2, 2.01, 3, 2, 1, 1])))
