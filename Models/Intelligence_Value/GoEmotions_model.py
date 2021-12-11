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
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import seaborn as sns

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

#%% Pre-process 

# Make all text lowercase
tweets['text'] = [word.lower() for word in tweets['text']]

# Remove all punctuation from text
for i in (range(0, len(tweets))):
    tweets['text'][i] = re.sub(r'[^\w\s]', '', tweets['text'][i])
print('Punctuation removed.')
    
# Remove numbers
for i in (range(0, len(GoEmotions))):
    GoEmotions['text'][i] = re.sub(r'[0-9]', '', GoEmotions['text'][i])
print('Numbers removed.')

# Load and configure stopwords
all_stopwords = stopwords.words('english')
additional_stopwords = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for letter in additional_stopwords:
    all_stopwords.append(letter)

# Remove stopwords
passwords = []
for sentence in range(0, len(tweets['text'])):
    review = tweets['text'][sentence].split()
    holding = []
    for word in review:
        if word not in all_stopwords:
            holding.append(word)
    passwords.append(' '.join(holding))

# Perform stemming on passwords
corpus = []
for sentence in range(0, len(passwords)):
    review = passwords[sentence].split()
    ps = PorterStemmer()
    holding = []
    for word in review:
        word = ps.stem(word)
        holding.append(word)
    corpus.append(' '.join(holding))

# What is the frequency distribution of each word? 
word_list = []
for sentence in range(0, len(corpus)):
    review = corpus[sentence].split()
    for word in review:
        word_list.append(word) # 48052 total words

# Generate frequency dictionary
freqDict =  dict()
visited = set()
for element in word_list:
    if element in visited:
        freqDict[element] = freqDict[element] + 1
    else:
        freqDict[element] = 1
        visited.add(element)
print('The number of unique words in the corpus is: ' + str(len(freqDict))) # 8723
word_freq = pd.Series(freqDict).sort_values(ascending=False)

# Visualise top 30 words
word_list1 = nltk.FreqDist(word_list)
word_list1 = pd.DataFrame({'Word':list(word_list1.keys()),
                             'Count':list(word_list1.values())})
g = word_list1.nlargest(columns="Count", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Word") 
ax.set(ylabel = 'Count') 
plt.show()

# Push corpus back into tweets dataframe & export to CSV
tweets['text'] = corpus
tweets.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets_corpus.csv")

#%% Vectorize tweet data
X_tweets = tfidf_vectorizer.fit_transform(corpus).toarray()

#%% Apply model

# Make predictions on available data
time_start = time.time()
y_pred_tweets_probs = clf.predict_proba(X_tweets)

# Lower threshold to increase volume of results
t = 0.2
y_pred_tweets = (y_pred_tweets_probs >= t).astype(int)
time_stop = time.time()
training_time = time_stop-time_start
print(f"Time to train: {training_time}") # 7 seconds

# Get name of predicted emotion for each row
emotions = pd.DataFrame(y_pred_tweets)
emotions.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\emotions.csv")
