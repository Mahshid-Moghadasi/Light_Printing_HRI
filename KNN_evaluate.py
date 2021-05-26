import json
import cv2
import time
import argparse
import os
import torch
import pandas as pd
import posenet
import json
from sklearn.neighbors import KNeighborsClassifier
## compute classification accuracy for the logistic regression model
from sklearn import metrics
from sklearn.model_selection import train_test_split

# import Matplotlib (scientific plotting library)
import matplotlib.pyplot as plt


## load training set
# with open('train_img.txt') as json_file:
with open('train_img.txt') as json_file:
    data = json.load(json_file)
    # print(data)

X = []
y = []
for v in data:
    X.append(v['coordinates'][10:22])
    y.append(v['label'])


## knn classifier

fig,_  = plt.subplots()
fig.set_size_inches(5.1667, 4.5167)
for ratio in [0.2, 0.4, 0.6, 0.8]:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= ratio, random_state=4)

    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    k_range = range(1, 10)
    scores = []
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        y_pred = knn.predict(X_test)
        scores.append(metrics.accuracy_score(y_test, y_pred))

    plt.plot(k_range, scores, label = 'test_size: T= {}%'.format(ratio*100) )
    plt.xlabel('Value of K for KNN')
    plt.ylabel('Testing Accuracy')

plt.legend()
plt.savefig('knn_evaluation.png')
plt.show()

