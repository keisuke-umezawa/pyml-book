from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
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
    
    svm = SVC(kernel='rbf', random_state=0, gamma=0.2, C=1.0)
    svm.fit(X_train_std, y_train)

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    utils.plot_decision_regions(X=X_combined_std, y=y_combined,
            classifier=svm, test_idx=range(105, 150))

    plt.xlabel('petal length [starndardized]')
    plt.xlabel('petal width [starndardized]')
    plt.legend(loc='upper left')
    plt.show()
    

if __name__ == '__main__':
    main()
