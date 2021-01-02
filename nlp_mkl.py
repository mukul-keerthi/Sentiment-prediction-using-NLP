# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 21:43:41 2020

@author: Mukul
"""

# Natural language processing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3) #quoting = 3 to ignore "" while processing the dataset

# Cleaning the texts
import re
import nltk #NLP toolkit
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import portStemmer #Stemming class - Take the root of the word to indicate what it means
corpus = [] #Contains List of words after stemming
for i in range (0,1000):
     #Replace anything other than alphabets with space, review being column name
    review = review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i]) 
    #lower() method returns a string where all characters are lower case.
    review = review.lower() 
    review = review.split()
    ps = PorterStemmer() #Porter stemmer object
    all_stopwords = stopwords.words('english')
    all_stopwords.remove('not') #Remove not from the list
    review = [ps.stem(word) for word in review if not word in set()] #Goes through all the words in review library except stopwords like as if etc..
    review = ' '.join(review) #Join to get the original string
    corpus.append(review) #Add review to the list
    
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values #Taking the last column

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
accuracy_score(y_test, y_pred)
