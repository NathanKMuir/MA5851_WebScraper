# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 14:21:30 2021

@author: natha
"""
# This script trains the model for predicting emotion on the GoEmotions dataset.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestClassifier # modelling
from sklearn.multiclass import OneVsRestClassifier # for multiclass modelling
from sklearn.metrics import accuracy_score, f1_score, precision_score
import time

# Import data and reset to previous state
GoEmotions = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\GoEmotions.csv")
del GoEmotions['Unnamed: 0']

# Remove NA rows
GoEmotions = GoEmotions.dropna()
GoEmotions = GoEmotions[0:10000]
GoEmotions = GoEmotions.reset_index(drop=True)

# Choosing hyperparameters - what is the number of unique words with an occurence of more than five?
word_list = []
corpus = list(GoEmotions['text'])

for i in range(0, len(corpus)):
    review = corpus[i].split()
    for word in review:
        word_list.append(word) # 14699 words

# Generate frequency dictionary
freqDict =  dict()
visited = set()
for element in word_list:
    if element in visited:
        freqDict[element] = freqDict[element] + 1
    else:
        freqDict[element] = 1
        visited.add(element)
newDict = { key:value for (key,value) in freqDict.items() if value > 3}
print(len(newDict)) # 10580 - use this as minimum words hyperparameter

# Vectorization - implementing a TFIDF approach
y = GoEmotions.iloc[0:, 1:]
tfidf_vectorizer = TfidfVectorizer(max_features = 999)
X = tfidf_vectorizer.fit_transform(corpus).toarray()
print(X.shape) # 10000 x 999
print(y.shape) # 10000 x 28

# Get training and testing split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 1234)

# Start timer
time_start = time.time()

# Assign classfier - random forest
rf = RandomForestClassifier(criterion="entropy", random_state=1234)
clf = OneVsRestClassifier(rf)

# Fit model on training data
clf.fit(X_train, y_train)

# Make predictions on training data & generate accuracy metrics
y_pred_train  = clf.predict(X_train)

# Stop timer
time_stop = time.time()
training_time = time_stop-time_start
print(f"Time to train: {training_time}")

# Accuracy metrics - performance on training data 
f1r1 = f1_score(y_train, y_pred_train, average="micro") 
print(f'F1 Score: {f1r1}') # 93.64%
psr1 = precision_score(y_train, y_pred_train, average="micro")
print(f'Precision Score: {psr1}') # 98.30% 
asr1 = accuracy_score(y_train, y_pred_train)
print(f'Accuracy Score: {asr1}') # 88.68% 

# Make predictions on test data & generate accuracy metrics
y_pred_test  = clf.predict(X_test)

# Accuracy metrics - performance on training data 
f1r2 = f1_score(y_test, y_pred_test, average="micro") 
print(f'F1 Score: {f1r2}') # 22.30%
psr2 = precision_score(y_test, y_pred_test, average="micro")
print(f'Precision Score: {psr2}') # 48.50% 
asr2 = accuracy_score(y_test, y_pred_test)
print(f'Accuracy Score: {asr2}') # 12.60% 

# Lower threshold
t = 0.3 
y_pred_new = (y_pred_train >= t).astype(int)
f1r1_1 = f1_score(y_train, y_pred_new, average="micro") # 65.80% 
psr1_1 = precision_score(y_train, y_pred_new, average="micro") # 74.73%
asr1_1 = accuracy_score(y_train, y_pred_new) # 55.67%