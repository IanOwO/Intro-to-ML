# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC # This is the SVM classifier class from sklearn.

# The two global variables below are the hyperparameters of the SVM with polynomial and rbf kernels.
# You should tune them to achieve the best performance on the test set.
degree_ = 3 # degree_ should be an integer.
gamma_ = 2

# Compute and return the Gram matrix of the given data.
def gram_matrix(X1, X2, kernel_function):
    return kernel_function(X1,X2)

# Compute and return the linear kernel between two vectors.
def linear_kernel(x1, x2):
    return np.dot(x1,x2.T)

# Compute and return the polynomial kernel between two vectors.
def polynomial_kernel(x1, x2, degree=degree_):
    return (np.dot(x1,x2.T) + 1)**degree

# Compute and return the rbf kernel between two vectors.  
def rbf_kernel(x1, x2, gamma=gamma_):
    tmp = np.zeros((x1.shape[0],x2.shape[0]))
    for i, item1 in enumerate(x1):
        for j, item2 in enumerate(x2):
           tmp[i][j] = np.exp(-1*(np.linalg.norm(item1 - item2) ** 2) * gamma) 
    return tmp

# Do not modify the main function architecture.
# You can only modify the hyperparameter C of the SVC class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]

    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    X_train = StandardScaler().fit_transform(X_train)
    X_test = StandardScaler().fit_transform(X_test)

    # Tune the hyperparameter (C) of the SVM with your linear kernel here.
    C_ = 0.01
    svc = SVC(kernel='precomputed', C=C_)
    svc.fit(gram_matrix(X_train, X_train, linear_kernel), y_train)
    y_pred = svc.predict(gram_matrix(X_test, X_train, linear_kernel))
    print(f"Accuracy of using linear kernel (C = {C_}): ", accuracy_score(y_test, y_pred))
    assert accuracy_score(y_test, y_pred) > 0.8

    # Tune the hyperparameter (degree) by the global variable degree_ of your polynomial_kernel function.
    # Tune the hyperparameter (C) of the SVM with your polynomial kernel here.
    C_ = 1
    svc = SVC(kernel='precomputed', C=C_)
    svc.fit(gram_matrix(X_train, X_train, polynomial_kernel), y_train)
    y_pred = svc.predict(gram_matrix(X_test, X_train, polynomial_kernel))
    print(f"Accuracy of using polynomial kernel (C = {C_}, degree = {degree_}): ", accuracy_score(y_test, y_pred))

    # Tune the hyperparameter (gamma) by the global variable gamma_ of your rbf_kernel function.
    # Tune the hyperparameter (C) of the SVM with your rbf kernel here.
    C_ = 1
    svc = SVC(kernel='precomputed', C=C_)
    svc.fit(gram_matrix(X_train, X_train, rbf_kernel), y_train)
    y_pred = svc.predict(gram_matrix(X_test, X_train, rbf_kernel))
    print(f"Accuracy of using rbf kernel (C = {C_}, gamma = {gamma_}): ", accuracy_score(y_test, y_pred))