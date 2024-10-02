# You are not allowed to import any additional packages/libraries.
import numpy as np
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# This function computes the gini impurity of a label array.
def gini(y):
    if np.shape(y)[0] == 0:
        return 0
    avg=np.average(y)
    return 1 - avg**2 - (1 - avg)**2

# This function computes the entropy of a label array.
def entropy(y):
    if np.shape(y)[0] == 0:
        return 0 
    avg=np.average(y)
    if avg == 1 or avg == 0:
        return 0
    else:
        return -avg * np.log2(avg) - (1 - avg) * np.log2(1 - avg)

class node:
    def __init__(self, ):
        self.left = None
        self.right = None
        self.threshold = None
        self.feature = None
        self.decision = None

# The decision tree classifier class.
# Tips: You may need another node class and build the decision tree recursively.
class DecisionTree():
    def __init__(self, criterion='gini', max_depth=None, rand=0):
        self.criterion = criterion
        self.max_depth = max_depth
        self.rand = rand
        self.tree = node() 
        self.col = np.zeros(6)
    
    # This function computes the impurity based on the criterion.
    def impurity(self, y):
        if self.criterion == 'gini':
            return gini(y)
        elif self.criterion == 'entropy':
            return entropy(y)
    
    def create_tree(self, X_data, y_data, cur_node, depth):
        threshold = None
        min_imp = 10
        feature=None
        left_dec = None
        right_dec = None
        leftidx=None
        rightidx=None
        
        if self.rand == 0:
            for fea in range(X_data.shape[1]):
                col=X_data[:, fea]
                for j in range(X_data.shape[0]):
                    tmp_left=np.argwhere(col <= X_data[j][fea]).flatten()
                    tmp_right=np.argwhere(col > X_data[j][fea]).flatten()
                        
                    tmp_imp_l=self.impurity(y_data[tmp_left])
                    tmp_imp_r=self.impurity(y_data[tmp_right])

                    if((tmp_imp_l * tmp_left.shape[0] + tmp_imp_r * tmp_right.shape[0]) / X_data.shape[0] < min_imp):
                        min_imp=(tmp_imp_l * tmp_left.shape[0] + tmp_imp_r * tmp_right.shape[0]) / X_data.shape[0]
                        threshold=X_data[j][fea]
                        feature=fea
                        leftidx=tmp_left
                        rightidx=tmp_right
                        if len(leftidx) == 0:
                            left_dec = None
                            if np.average(y_data[rightidx]) >= 0.5:                        
                                right_dec=1
                            else:
                                right_dec=0
                        elif len(rightidx) == 0:
                            right_dec = None
                            if np.average(y_data[leftidx]) > 0.5:                        
                                left_dec=1
                            else:
                                left_dec=0
                        else:
                            if np.average(y_data[leftidx]) >= 0.5:                        
                                left_dec=1
                            else:
                                left_dec=0
                            if np.average(y_data[rightidx]) > 0.5:                        
                                right_dec=1
                            else:
                                right_dec=0
        else:
            feature = np.random.randint(X_data.shape[1])
            # print(feature)
            threshold = np.random.choice(X_data[:,feature],1)
            # print(threshold)
            tmp_left=np.argwhere(X_data[:,feature] <= threshold).flatten()
            tmp_right=np.argwhere(X_data[:,feature] > threshold).flatten()
            leftidx=tmp_left
            rightidx=tmp_right
            if len(leftidx) == 0:
                left_dec = None
                if np.average(y_data[rightidx]) >= 0.5:                        
                    right_dec=1
                else:
                    right_dec=0
            elif len(rightidx) == 0:
                right_dec = None
                if np.average(y_data[leftidx]) > 0.5:                        
                    left_dec=1
                else:
                    left_dec=0
            else:
                if np.average(y_data[leftidx]) >= 0.5:                        
                    left_dec=1
                else:
                    left_dec=0
                if np.average(y_data[rightidx]) > 0.5:                        
                    right_dec=1
                else:
                    right_dec=0
  
        cur_node.threshold=threshold
        cur_node.feature=feature
        cur_node.left = node()
        cur_node.right = node() 
        self.col[feature] += 1
        if min_imp == 0 or depth + 1 >= self.max_depth:
            cur_node.left.decision = left_dec
            cur_node.right.decision = right_dec
            return
        else:
            self.create_tree(X_data[leftidx], y_data[leftidx], cur_node.left, depth + 1)
            self.create_tree(X_data[rightidx], y_data[rightidx], cur_node.right, depth + 1)
            
            

    # This function fits the given data using the decision tree algorithm.
    def fit(self, X, y):
        cur = self.tree
        self.create_tree(X, y, cur, 0)
        if self.rand == 0 and self.max_depth == 15:
            self.plot_feature_importance_img(self.col)
    
    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred_val=[]
        for features in X:
            cur_node = self.tree
            while True:
                if cur_node.threshold == None:
                    break
                elif features[cur_node.feature] > cur_node.threshold:
                    cur_node = cur_node.right
                elif features[cur_node.feature] <= cur_node.threshold:
                    cur_node = cur_node.left
            pred_val.append(cur_node.decision)
        return pred_val
    
    # This function plots the feature importance of the decision tree.
    def plot_feature_importance_img(self, columns):
        fig = plt.figure()
        plt.title(self.max_depth)
        plt.barh(["age","sex","cp","fbs","thalach","thal"], self.col, color='blue')
        plt.show()

# The AdaBoost classifier class.
class AdaBoost():
    def __init__(self, criterion='gini', n_estimators=200):
        self.criterion = criterion 
        self.n_estimators = n_estimators

        self.model = []
        self.alpha = []

        
    # This function fits the given data using the AdaBoost algorithm.
    # You need to create a decision tree classifier with max_depth = 1 in each iteration.
    def fit(self, X, y):
        sample = X.shape[0]
        weight = np.ones(sample) / sample
        # y[y == 0] = -1
        for i in range(self.n_estimators):
            model=DecisionTree(criterion=self.criterion, max_depth=1, rand=1)
            model.fit(X,y)
            pred = model.predict(X)
            err = np.sum(weight * (pred != y))
            # print(err)
            alpha=0.5 * np.log((1 - err)/err)
            weight *= np.exp(-alpha * y * pred)
            weight /= np.sum(weight)
            self.alpha.append(alpha)
            self.model.append(model)

    # This function takes the input data X and predicts the class label y according to your trained model.
    def predict(self, X):
        pred = np.array([model.predict(X) for model in self.model])
        weighted = np.dot(self.alpha, pred)
        weighted[weighted > 0] = 1
        weighted[weighted <= 0] = 0
        return weighted
        

# Do not modify the main function architecture.
# You can only modify the value of the random seed and the the arguments of your Adaboost class.
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

# Set random seed to make sure you get the same result every time.
# You can change the random seed if you want to.
    np.random.seed(0)

# Decision Tree
    print("Part 1: Decision Tree")
    data = np.array([0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1])
    print(f"gini of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {gini(data)}")
    print(f"entropy of [0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1]: {entropy(data)}")
    tree = DecisionTree(criterion='gini', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (gini with max_depth=7):", accuracy_score(y_test, y_pred))
    tree = DecisionTree(criterion='entropy', max_depth=7)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    print("Accuracy (entropy with max_depth=7):", accuracy_score(y_test, y_pred))

    tree = DecisionTree(criterion='gini', max_depth=15)
    tree.fit(X_train, y_train)
    y_pred = tree.predict(X_test)
    

# AdaBoost
    print("Part 2: AdaBoost")
    # Tune the arguments of AdaBoost to achieve higher accuracy than your Decision Tree.
    ada = AdaBoost(criterion='gini', n_estimators=15)
    ada.fit(X_train, y_train)
    y_pred = ada.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))


    
