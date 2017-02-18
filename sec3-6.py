from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
from perceptron import utils as utils

def main():
    iris = datasets.load_iris()
    
    X = iris.data[:, [2, 3]]
    
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
    
    sc = StandardScaler()
    
    sc.fit(X_train)
    
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    tree = RandomForestClassifier(criterion='entropy', n_estimators=10, random_state=1, n_jobs=2)
    tree.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    utils.plot_decision_regions(X=X_combined_std, y=y_combined,
            classifier=tree, test_idx=range(105, 150))

    plt.xlabel('petal length [starndardized]')
    plt.xlabel('petal width [starndardized]')
    plt.legend(loc='upper left')
    plt.show()


if __name__ == '__main__':
    main()
