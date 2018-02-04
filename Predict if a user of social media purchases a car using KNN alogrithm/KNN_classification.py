# -*- coding: utf-8 -*-
"""

@author: VIPUL
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("C:/Users/VIPUL/Dropbox/Interview preparations/full time resumes/files to be added to github/Machine Learning/Predict if a user of social media purchases a car using KNN alogrithm/Social_Network_Ads.csv")

X = dataset.iloc[:,[2,3]].values
y = dataset.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

 from sklearn.neighbors import KNeighborsClassifier
 classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
 classifier.fit(X_train, y_train)
 
 y_pred = classifier.predict(X_test)
 
 from sklearn.metrics import confusion_matrix
 cm = confusion_matrix(y_test, y_pred)
 
 (55+21)/80
 
 