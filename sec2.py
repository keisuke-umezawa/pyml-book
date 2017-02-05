import pandas as pd
from perceptron import models as perceptron

df = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/iris/iris.data', header=None)
print(df.tail())

import matplotlib.pyplot as plt
import numpy as np

# select setosa and versicolar
y = df.iloc[0:100, 4].values
y = np.where(y == 'Iris-setosa', -1, 1)

# extract sepal length and petal length
X = df.iloc[0:100, [0, 2]].values

# plot data
plt.scatter(X[:50, 0], X[:50, 1], color='red', marker='o', label='setosa')
plt.scatter(X[50:100, 0], X[50:100, 1], color='blue', marker='x', label='versicolor')

plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [xm]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./images/02_06.png', dpi=300)
plt.show()

ppn = perceptron.Perceptron(eta=0.1, n_iter=10)

ppn.fit(X, y)

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')
plt.xlabel('Epochs')
plt.ylabel('Number of updates')

plt.tight_layout()
plt.savefig('./images/perceptron_1.png', dpi=300)
plt.show()


from perceptron import utils as utils


utils.plot_decision_regions(X, y, classifier=ppn)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.legend(loc='upper left')

plt.tight_layout()
plt.savefig('./perceptron_2.png', dpi=300)
plt.show()


