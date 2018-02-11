# -*- coding: utf-8 -*-

# importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# importing dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

# creating an empty list to fill with the words
corpus  = []

# running loop to the number of reviews
for j in range(1000):
    # just considering the letters and replacing everthing else by a space
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][j])
    
    # make every thing lower case
    review = review.lower()
    
    # splitting it and making it a list of words
    review = review.split()
    
    # creating an instance of a class which is used to stem the words
    ps = PorterStemmer()
    
    # using set to get the words faster and then stemming those words if not in stopwords and then joining those words
    words = [ps.stem(i) for i in review if i not  in set(stopwords.words('english'))]
    words = ' '.join(words)
    
    
    # finally add the string to a list 
    corpus.append(words)
    

# countvectorizer class to make a sparse matrix of words with their counts

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features= 1500)
X = cv.fit_transform(corpus).toarray()
Y =  dataset.iloc[:,1].values

# splitting the data
from sklearn.cross_validation import train_test_split
X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size = 0.2, random_state = 0)


# running naive bayes classifier over the model
from sklearn.naive_bayes import GaussianNB
gauss = GaussianNB()
gauss.fit(X_tr,Y_tr)

# prediciting the reviews
y_pred = gauss.predict(X_ts)

# creating the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_ts, y_pred)


