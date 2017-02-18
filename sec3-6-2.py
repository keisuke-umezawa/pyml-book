
def gini(p):
    return p * (1 -p) + (1 - p) * (1 - (1 -p))

def entorpy(p):
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

def error(p):
    return 1 - np.max([p, 1 - p])



from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.tree import export_graphviz
import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from perceptron import utils as utils

def main():
    x = np.arange(0.0, 1.0, 0.01)
    
    ent = [entorpy(p) if p != 0 else None for p in x]
    sc_ent = [e * 0.5 if e else None for e in ent]
    err = [error(p) for p in x]
    
    fig = plt.figure()
    ax = plt.subplot(111)
    for i, lab, ls, c, in zip([ent, sc_ent, gini(x), err],
                              ['Entropy', 'Entropy (scaled)',
                               'Gini Impurity', 'Misclassification Error'],
                              ['-', '-', '--', '-.'],
                              ['black', 'lightgray', 'red', 'green', 'cyan']):
        line = ax.plot(x, i, label=lab, linestyle=ls, lw=2, color=c)
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
            ncol=3, fancybox=True, shadow=False)
    ax.axhline(y=0.5, linewidth=1, color='k', linestyle='--')
    ax.axhline(y=1.0, linewidth=1, color='k', linestyle='--')
    plt.ylim([0, 1.1])
    plt.xlabel('p(i=1)')
    plt.ylabel('Impurity Index')
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
    
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
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
