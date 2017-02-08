from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from perceptron import utils as utils
from perceptron import sigmoid as sig

def main():
    z = np.arange(-7, 7, 0.1)
    phi_z = sig.sigmoid(z)
    plt.plot(z, phi_z)
    plt.axvline(0.0, color='k')
    plt.ylim(-0.1, 1.1)
    plt.xlabel('z')
    plt.ylabel('$\phi (z)$')

    # y axix ticks and gridline
    plt.yticks([0.0, 0.5, 1.0])
    ax = plt.gca()
    ax.yaxis.grid(True)

    plt.tight_layout()
    plt.show()

    iris = datasets.load_iris()
    
    X = iris.data[:, [2, 3]]
    
    y = iris.target
    
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=0)
    
    sc = StandardScaler()
    
    sc.fit(X_train)
    
    X_train_std = sc.transform(X_train)
    X_test_std = sc.transform(X_test)
    
    lr = LogisticRegression(C=1000.0, random_state=0)
    lr.fit(X_train_std, y_train)

    y_pred = lr.predict(X_test_std)

    print('Mixclassfied samples: %d' % (y_test != y_pred).sum())
    print('Accurasy: %.2f' % accuracy_score(y_test, y_pred))

    X_combined_std = np.vstack((X_train_std, X_test_std))
    y_combined = np.hstack((y_train, y_test))

    utils.plot_decision_regions(X=X_combined_std, y=y_combined,
            classifier=lr, test_idx=range(105, 150))

    plt.xlabel('petal length [starndardized]')
    plt.xlabel('petal width [starndardized]')
    plt.legend(loc='upper left')
    plt.show()
    

if __name__ == '__main__':
    main()
