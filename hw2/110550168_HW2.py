# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

class LogisticRegression:
    def __init__(self, learning_rate=0.1, iteration=100):
        self.learning_rate = learning_rate
        self.iteration = iteration
        self.weights = None
        self.intercept = None

    # This function computes the gradient descent solution of logistic regression.
    def fit(self, X, y):
        cal_matrix = np.ones(7)
        N = X.shape[0]
        X_add_one_col = np.c_[np.ones(N),X]

        for i in range(self.iteration):
            loss = y - self.sigmoid(np.dot(X_add_one_col,cal_matrix.T))
            pred = loss * (-1/N)
            grad = np.dot(X_add_one_col.T,pred)
            cal_matrix += (-self.learning_rate) * grad.T

            # if i % 1 == 0:
            #     loss_table.append(np.mean(np.power(loss,2)))
            #     print("i: ",i,"  loss: ",-np.mean(y*np.log(self.sigmoid(np.dot(X_add_one_col,cal_matrix.T))) + (1-y)*np.log(1-self.sigmoid(np.dot(X_add_one_col,cal_matrix.T)))))

        self.weights = cal_matrix[1:]
        self.intercept = cal_matrix[0]
            
    # This function takes the input data X and predicts the class label y according to your solution.
    def predict(self, X):
        z = np.dot(self.weights ,X.T) + self.intercept
        pred = self.sigmoid(z)
        pred[pred > 0.5] = 1
        pred[pred < 0.5] = 0
        return pred

    # This function computes the value of the sigmoid function.
    def sigmoid(self, x):
        return 1/(1+np.exp(-x))
        

class FLD:
    def __init__(self):
        self.w = None
        self.m0 = None
        self.m1 = None
        self.sw = None
        self.sb = None
        self.slope = None

    # This function computes the solution of Fisher's Linear Discriminant.
    def fit(self, X, y):
        list0 = np.where(y == 0)
        list1 = np.where(y == 1)
        mat0 = X[list0]
        mat1 = X[list1]
        self.m0 = np.mean(X[list0],axis=0,keepdims=True)
        # print("------------",self.m0)
        self.m1 = np.mean(X[list1],axis=0,keepdims=True)
        # print("mat0:",mat0)
        # print("meanmmmmmmmmmmm:",self.m0)
        mat0 = np.subtract(mat0,self.m0)
        mat1 = np.subtract(mat1,self.m1)
        # print("mat0 stape:",np.shape(mat0))
        # print("after:",mat0)
        self.sb = np.dot((self.m1-self.m0).T,(self.m1-self.m0))
        # print("sb size:",np.shape(self.sb))
        self.sw = np.dot(mat0.T,mat0) + np.dot(mat1.T,mat1)

        self.w = np.dot(np.linalg.inv(self.sw),(self.m1-self.m0).T).T
        self.w = self.w / np.linalg.norm(self.w)
        # print("self.w:",self.w)
        self.slope = self.w[0][1] / self.w[0][0]
        # print("slope:",self.slope)

        # intercept = 20
        # line_x = np.linspace(-55, -5, 10)
        # line_y = self.slope * line_x + intercept
        # mat0_c1 = X[list0].T[0]
        # mat0_c2 = X[list0].T[1]
        # mat1_c1 = X[list1].T[0]
        # mat1_c2 = X[list1].T[1]
        # p1_x = (self.slope * mat0_c2 + mat0_c1 - self.slope * intercept) / (self.slope**2 + 1)
        # p1_y = (self.slope**2 * mat0_c2 + self.slope * mat0_c1 + intercept) / (self.slope**2 + 1)
        # p2_x = (self.slope * mat1_c2 + mat1_c1 - self.slope * intercept) / (self.slope**2 + 1)
        # p2_y = (self.slope**2 * mat1_c2 + self.slope * mat1_c1 + intercept) / (self.slope**2 + 1)
        # plt.title(f'Projection Line: w={self.slope}, b={intercept}')
        # plt.plot(line_x, line_y, c='black', linewidth=0.8)
        # plt.plot([mat1_c1, p2_x], [mat1_c2, p2_y], c='blue', linewidth=0.1)
        # plt.plot([mat0_c1, p1_x], [mat0_c2, p1_y], c='red', linewidth=0.1)
        # plt.plot(mat1_c1, mat1_c2, '.', c='blue', markersize=3)
        # plt.plot(mat0_c1, mat0_c2, '.', c='red', markersize=3)
        # plt.plot(p2_x, p2_y, '.', c='blue', markersize=1)
        # plt.plot(p1_x, p1_y, '.', c='red', markersize=1)
        # plt.show()



    # This function takes the input data X and predicts the class label y by comparing the distance between the projected result of the testing data with the projected means (of the two classes) of the training data.
    # If it is closer to the projected mean of class 0, predict it as class 0, otherwise, predict it as class 1.
    def predict(self, X):
        a = np.abs(np.dot(X,self.w.T) - np.dot(self.m0,self.w.T))
        b = np.abs(np.dot(X,self.w.T) - np.dot(self.m1,self.w.T))
        return np.greater(a,b).astype(int)
        

    # This function plots the projection line of the testing data.
    # You don't need to call this function in your submission, but you have to provide the screenshot of the plot in the report.
    def plot_projection(self, X):
        pass
     
# Do not modify the main function architecture.
# You can only modify the value of the arguments of your Logistic Regression class.
if __name__ == "__main__":
# Data Loading
    train_df = DataFrame(read_csv("train.csv"))
    test_df = DataFrame(read_csv("test.csv"))

# Part 1: Logistic Regression
    # Data Preparation
    # Using all the features for Logistic Regression
    X_train = train_df.drop(["target"], axis=1)
    y_train = train_df["target"]
    X_test = test_df.drop(["target"], axis=1)
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    LR = LogisticRegression(learning_rate=0.0002, iteration=300000)
    LR.fit(X_train, y_train)
    y_pred = LR.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 1: Logistic Regression")
    print(f"Weights: {LR.weights}, Intercept: {LR.intercept}")
    print(f"Accuracy: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.75, "Accuracy of Logistic Regression should be greater than 0.75"

# Part 2: Fisher's Linear Discriminant
    # Data Preparation
    # Only using two features for FLD
    X_train = train_df[["age", "thalach"]]
    y_train = train_df["target"]
    X_test = test_df[["age", "thalach"]]
    y_test = test_df["target"]
    
    X_train = X_train.to_numpy()
    y_train = y_train.to_numpy()
    X_test = X_test.to_numpy()
    y_test = y_test.to_numpy()

    # Model Training and Testing
    FLD = FLD()
    FLD.fit(X_train, y_train)
    y_pred = FLD.predict(X_test)
    accuracy = accuracy_score(y_test , y_pred)
    print(f"Part 2: Fisher's Linear Discriminant")
    print(f"Class Mean 0: {FLD.m0}, Class Mean 1: {FLD.m1}")
    print(f"With-in class scatter matrix:\n{FLD.sw}")
    print(f"Between class scatter matrix:\n{FLD.sb}")
    print(f"w:\n{FLD.w}")
    print(f"Accuracy of FLD: {accuracy}")
    # You must pass this assertion in order to get full score for this part.
    assert accuracy > 0.65, "Accuracy of FLD should be greater than 0.65"

