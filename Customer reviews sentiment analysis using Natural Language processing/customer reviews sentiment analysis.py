# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 12:35:18 2017

@author: VIPUL
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
#import
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

corpus = []
for i in range(0,1000):
    #first step of the cleaning process
    review = re.sub('[^a-zA-Z]',' ',dataset.loc[i,'Review'])
    #second step
    review = review.lower()
    #third step - remove irrelevant words and fourth step - stemming.. keeping only the root of the word
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english')) ]
    #join the words and make a string separated by space
    review = ' '.join(review)
    corpus.append(review)
    
#creating the bag of words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,1].values
#splitting into training and testing
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y ,  test_size = 0.20 , random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#fitting naive bayes to the training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

#predicting the test set results
y_pred = classifier.predict(X_test)

#making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
(54 + 87)/200