# Data
# [height, weight, shoe_size]
X = [
    [181, 80, 44],
    [177, 70, 43],
    [160, 60, 38],
    [154, 54, 37],
    [166, 65, 40],
    [190, 90, 47],
    [175, 64, 39],
    [177, 70, 40],
    [159, 55, 37],
    [171, 75, 42],
    [181, 85, 43],
]

Y = [
    "male",
    "male",
    "female",
    "female",
    "male",
    "male",
    "female",
    "female",
    "female",
    "male",
    "male",
]

from sklearn import tree

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
prediction = clf.predict([[190, 70, 43]])
print(f"DecisionTreeClassifier {prediction}")

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb = gnb.fit(X, Y)
prediction = gnb.predict([[190, 70, 43]])
print(f"GaussianNB {prediction}")

from sklearn.linear_model import LogisticRegression

neigh = LogisticRegression()
neigh.fit(X, Y)
prediction = neigh.predict([[190, 70, 43]])
print(f"logistic regression {prediction}")

from sklearn.neighbors import KNeighborsClassifier

neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X, Y)
prediction = neigh.predict([[190, 70, 43]])
print(f"KNeighborsClassifier {prediction}")
