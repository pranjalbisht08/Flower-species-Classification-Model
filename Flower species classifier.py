# Loading required modules
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

#Loading dataset
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
species = iris.target_names

#training the classifier
clf = KNeighborsClassifier()
clf.fit(X, y)
preds = clf.predict(X)

#defining color for each species
colors = ['purple', 'green', 'yellow']

#scatter plot
for i in range(3):
    plt.scatter(X[preds == i, 0],
                X[preds == i, 1],
                c=colors[i],
                label=species[i])

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.title("KNN Classification - Flower Species")
plt.legend()
plt.show()