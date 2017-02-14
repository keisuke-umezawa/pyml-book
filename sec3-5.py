from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from perceptron import utils as utils

def main():
    np.random.seed(0)

    X_xor = np.random.randn(200, 2)

    y_xor = np.logical_xor(X_xor[:, 0] > 0, X_xor[:, 1] > 0)
    y_xor = np.where(y_xor, 1, -1)

    plt.scatter(X_xor[y_xor==1, 0], X_xor[y_xor==1, 1],
            c='b', marker='x', label='1')
    plt.scatter(X_xor[y_xor==-1, 0], X_xor[y_xor==-1, 1],
            c='r', marker='s', label='-1')
    plt.xlim([-3, 3])
    plt.ylim([-3, 3])
    plt.legend(loc='best')
    plt.show()
    
    svm = SVC(kernel='rbf', random_state=0, gamma=0.1, C=10.0)
    svm.fit(X_xor, y_xor)

    utils.plot_decision_regions(X=X_xor, y=y_xor, classifier=svm)

    plt.legend(loc='upper left')
    plt.show()
    

if __name__ == '__main__':
    main()
