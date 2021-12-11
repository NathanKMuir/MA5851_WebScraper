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
import matplotlib.pyplot as plt

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
print(f"Time to train: {training_time}") # 268 seconds

# Accuracy metrics - performance on training data 
f1r1 = f1_score(y_train, y_pred_train, average="micro") 
print(f'F1 Score: {f1r1}') # 93.64%
psr1 = precision_score(y_train, y_pred_train, average="micro")
print(f'Precision Score: {psr1}') # 98.30% 
asr1 = accuracy_score(y_train, y_pred_train)
print(f'Accuracy Score: {asr1}') # 88.68% 

# Make predictions on test data & generate accuracy metrics
y_pred_test  = clf.predict(X_test)
y_pred_test_probs = clf.predict_proba(X_test)

# Accuracy metrics - performance on training data 
f1r2 = f1_score(y_test, y_pred_test, average="micro") 
print(f'F1 Score: {f1r2}') # 22.30%
psr2 = precision_score(y_test, y_pred_test, average="micro")
print(f'Precision Score: {psr2}') # 48.50% 
asr2 = accuracy_score(y_test, y_pred_test)
print(f'Accuracy Score: {asr2}') # 12.60% 

# Lower threshold to increase volume of results
t = 0.2
y_pred_new = (y_pred_test_probs >= t).astype(int)
f1r3 = f1_score(y_test, y_pred_new, average="micro") 
print(f'F1 Score: {f1r3}') # 34.80%
psr3 = precision_score(y_test, y_pred_new, average="micro")
print(f'Precision Score: {psr3}') # 33.57% 
asr3 = accuracy_score(y_test, y_pred_new)
print(f'Accuracy Score: {asr3}') # 19.40% 

# Adjust hyperparameters 1 and maintain lower threshold
time_start = time.time()
rf = RandomForestClassifier(criterion="entropy", random_state=1234, n_estimators=50, max_features=50)
clf2 = OneVsRestClassifier(rf)
clf2.fit(X_train, y_train)
y_pred_test_probs2 = clf2.predict_proba(X_test)
time_stop = time.time()
training_time = time_stop-time_start
print(f"Time to train: {training_time}") # 204 seconds

t = 0.2
y_pred_new2 = (y_pred_test_probs2 >= t).astype(int)
f1r4 = f1_score(y_test, y_pred_new2, average="micro") 
print(f'F1 Score: {f1r4}') # 34.10%
psr4 = precision_score(y_test, y_pred_new2, average="micro")
print(f'Precision Score: {psr4}') # 32.19% 
asr4 = accuracy_score(y_test, y_pred_new2)
print(f'Accuracy Score: {asr4}') # 19.16%

# Adjust hyperparameters 2 and maintain lower threshold
time_start = time.time()
rf = RandomForestClassifier(criterion="entropy", random_state=1234, n_estimators=250, max_features=150)
clf3 = OneVsRestClassifier(rf)
clf3.fit(X_train, y_train)
y_pred_test_probs3 = clf3.predict_proba(X_test)
time_stop = time.time()
training_time = time_stop-time_start
print(f"Time to train: {training_time}") # 2665 seconds

t = 0.2
y_pred_new3 = (y_pred_test_probs3 >= t).astype(int)
f1r5 = f1_score(y_test, y_pred_new3, average="micro") 
print(f'F1 Score: {f1r5}') # 34.22%
psr5 = precision_score(y_test, y_pred_new3, average="micro")
print(f'Precision Score: {psr5}') # 32.32% 
asr5 = accuracy_score(y_test, y_pred_new3)
print(f'Accuracy Score: {asr5}') # 18.80%

# Plot F1 score comparisons
data_dict = {'Model 1': f1r2, 'Model 2': f1r3,
             'Model 3': f1r4, 'Model 4': f1r5}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.title("F1 Score Across All Models")
plt.show()

# Plot Precision Score score comparisons
data_dict = {'Model 1': psr2, 'Model 2': psr3,
             'Model 3': psr4, 'Model 4': psr5}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Models")
plt.ylabel("Precision Score")
plt.title("Precision Score Across All Models")
plt.show()

# Plot Accuracy Score score comparisons
data_dict = {'Model 1': asr2, 'Model 2': asr3,
             'Model 3': asr4, 'Model 4': asr5}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Models")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Score Across All Models")
plt.show() # Model 2 (y_pred_new) is the highest performing

#%% APPLY TO TWEETS 
# Load in tweets data
tweets = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets.csv")
tweets['date'] = pd.to_datetime(tweets['date'])
del tweets['Unnamed: 0']

# Pre-process ()