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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=4)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(metrics.accuracy_score(y_test, y_pred))

