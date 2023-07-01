import pandas as pd
import numpy as np

import sklearn
from sklearn.model_selection import train_test_split
from sklearn import linear_model, preprocessing

import matplotlib.pyplot as plt
import pickle
from matplotlib import style

data = pd.read_csv('garments_worker_productivity.csv', sep=",")
data = data[["targeted_productivity", "actual_productivity", "incentive", "over_time", "idle_time"]]

max_over_time = data[["over_time"]].max()
min_over_time = data[["over_time"]].min()

max_incentive = data[["incentive"]].max()
min_incentive = data[["incentive"]].min()

data["incentive"] = (data[["incentive"]] - min_incentive) / (max_incentive - min_incentive)
data["over_time"] = (data[["over_time"]] - min_over_time) / (max_over_time - min_over_time)

predict = "targeted_productivity"

X = np.array(data.drop([predict], axis=1))
y = np.array(data[predict])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
"""
best = 0
for _ in range(50):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
    model = linear_model.LinearRegression()

    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)

    if acc > best:
        best = acc
        with open("garment_worker_model.pickle", "wb") as f:
            pickle.dump(model, f)
"""
pickle_in = open("garment_worker_model.pickle", "rb")
model = pickle.load(pickle_in)

print("coefficients: \n", model.coef_)
print("intercept: \n", model.intercept_)

y_pred = model.predict(X_test)

for x in range(len(y_pred)):
    print("predicted value: ", y_pred[x], "actual value: ", y_test[x])