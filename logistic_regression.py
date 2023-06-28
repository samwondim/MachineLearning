import sklearn
from sklearn.utils import shuffle
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model, preprocessing

import pandas as pd
import numpy as np

data = pd.read_csv("car.data")

le = preprocessing.LabelEncoder() # lets us convert none numerical data into numerical data.

buying = le.fit_transform(list(data["buying"]))
maint = le.fit_transform(list(data["maint"]))
door = le.fit_transform(list(data["door"]))
persons = le.fit_transform(list(data["persons"]))
cls = le.fit_transform(list(data["class"]))
safety = le.fit_transform(list(data["safety"]))

predict = "class"

X = list(zip(buying, maint, door, persons, cls, safety)) # zip the columns to one list
y = list(cls)

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size = .1)

model = KNeighborsClassifier(n_neighbors=5)

model.fit(x_train, y_train)
acc = model.score(x_test, y_test)

print(acc)

predicted = model.predict(x_test)
# see what the datapoints are
names = ["unacc", "acc", "good", "vgood"]

for x in range(len(x_test)):
    print("predicted: ", names[predicted[x]], "data: ", x_test[x], "actual: ", names[y_test[x]])
    n = model.kneighbors([x_test[x]], 5,True)
    print("N: ", n)