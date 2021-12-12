# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 00:50:56 2021

@author: natha
"""

# This script creates machine learning algorithms for the prediction of entities, and then creates a recommender system.

import pandas as pd
import re
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer # for mutliclass processing
from sklearn.multiclass import OneVsRestClassifier # for multiclass modelling
from sklearn.model_selection import train_test_split # for creation of test and train split
from sklearn.metrics import accuracy_score, f1_score, precision_score
from sklearn.linear_model import LogisticRegression # modelling
import time

# Import intel tweets dataframe
intel = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel_NER.csv")
intel['date'] = pd.to_datetime(intel['date'])
del intel['Unnamed: 0']

#%% Visualisations of entities 

# Visualise location
locations = list(intel['location'])
for i in (range(0, len(locations))):
    locations[i] = re.sub(r'[^\w\s]', '', locations[i])
word_list = []
for i in range(0, len(locations)):
    review = locations[i].split()
    for word in review:
        word_list.append(word)
all_locations = nltk.FreqDist(word_list)
unique_loc = len(all_locations)
print(f'The number of unique locations is: {unique_loc}') # 218
all_locations = pd.DataFrame({'location':list(all_locations.keys()),
                             'count':list(all_locations.values())})
g_loc = all_locations.nlargest(columns="count", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g_loc, x= "count", y = "location") 
ax.set(ylabel = 'count') 
plt.show()

# Visualise people
people = intel['person']
for i in (range(0, len(people))):
    people[i] = re.sub(r'[^\w\s]', '', people[i])
word_list = []
for i in range(0, len(people)):
    review = people[i].split()
    for word in review:
        word_list.append(word)
all_people = nltk.FreqDist(word_list)
unique_people = len(all_people)
print(f'The number of unique people is: {unique_people}') # 431
all_people = pd.DataFrame({'person':list(all_people.keys()),
                             'count':list(all_people.values())})
g_peo = all_people.nlargest(columns="count", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g_peo, x= "count", y = "person") 
ax.set(ylabel = 'count') 
plt.show()

# Visualise organisations
organisations = intel['organisation']
for i in (range(0, len(organisations))):
    organisations[i] = re.sub(r'[^\w\s]', '', organisations[i])
word_list = []
for i in range(0, len(organisations)):
    review = organisations[i].split()
    for word in review:
        word_list.append(word)
all_orgs = nltk.FreqDist(word_list)
unique_orgs = len(all_orgs)
print(f'The number of unique organisations is: {unique_orgs}') # 348
all_orgs = pd.DataFrame({'organisation':list(all_orgs.keys()),
                             'count':list(all_orgs.values())})
g_org = all_orgs.nlargest(columns="count", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g_org, x= "count", y = "organisation") 
ax.set(ylabel = 'count') 
plt.show()

#%% Pre-processing to create corpus from intel text

# Set aside copy of unclean tweets
unclean_corpus = intel['text']

# Make all text lowercase
intel['text'] = [word.lower() for word in intel['text']]

# Remove all punctuation from text
for i in (range(0, len(intel))):
    intel['text'][i] = re.sub(r'[^\w\s]', '', intel['text'][i])
print('Punctuation removed.')
    
# Remove numbers
for i in (range(0, len(intel))):
    intel['text'][i] = re.sub(r'[0-9]', '', intel['text'][i])
print('Numbers removed.')

# Load and configure stopwords
all_stopwords = stopwords.words('english')
additional_stopwords = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for letter in additional_stopwords:
    all_stopwords.append(letter)

# Remove stopwords
passwords = []
for sentence in range(0, len(intel['text'])):
    review = intel['text'][sentence].split()
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
        word_list.append(word) # 46599 total words

# Generate frequency dictionary
freqDict =  dict()
visited = set()
for element in word_list:
    if element in visited:
        freqDict[element] = freqDict[element] + 1
    else:
        freqDict[element] = 1
        visited.add(element)
print('The number of unique words in the corpus is: ' + str(len(freqDict))) # 7973
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

# Push corpus back into intel dataframe
intel['text'] = corpus

#%% Seperate intel into df containing location entities, and df containing organisation entities

intel_unclean = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel_NER.csv")
intel['location'] = intel_unclean['location']
intel['organisation'] = intel_unclean['organisation']

# Set empty entities as NA
for i in range(0, len(intel['location'])):
    if intel['location'][i] == '[]':
        intel['location'][i] = np.NaN
    else:
        pass
for i in range(0, len(intel['organisation'])):
    if intel['organisation'][i] == "[]":
        intel['organisation'][i] = np.NaN
    else:
        pass     

# Generate entity-specific datasets
intel_locations = intel.dropna(subset=['location'])
print(len(intel_locations)) # 881
intel_locations = intel_locations.reset_index(drop=True)
intel_organisations = intel.dropna(subset=['organisation'])
print(len(intel_organisations)) # 592
intel_organisations = intel_organisations.reset_index(drop=True)

#%% Vectorize corpus and prepare sets

# Create corpus for each subset
corpus_locations = intel_locations['text']
corpus_organisations = intel_organisations['text']

# Perform TFIDF on each dataframe
tfidf_vectorizer = TfidfVectorizer(max_features = 999)
X_loc = tfidf_vectorizer.fit_transform(corpus_locations).toarray()
X_org = tfidf_vectorizer.fit_transform(corpus_organisations).toarray()

# Get y values for locations
y_loc = []
locations = intel_locations['location']
for row in locations:
    temp = [row]
    y_loc.append(temp)

# Get y values for organisations
y_org = []
organisations = intel_organisations['organisation']
for row in organisations:
    temp = [row]
    y_org.append(temp)

#%% 

# Proceed with MLB for y_loc
multilabel_binarizer = MultiLabelBinarizer()
multilabel_binarizer.fit(y_loc)
y_loc = multilabel_binarizer.transform(y_loc)
print(X_loc.shape) # 1583 x 999
print(y_loc.shape) # 1583 x 687
print(y_loc[0])
print('This vector translates to: ' + str(multilabel_binarizer.inverse_transform(y_loc)[0])) # Yemen

# Proceed with MLB for y_org
multilabel_binarizer2 = MultiLabelBinarizer()
multilabel_binarizer2.fit(y_org)
y_org = multilabel_binarizer2.transform(y_org)
print(X_org.shape) # 1799 x 999
print(y_org.shape) # 1799 x 1106

# Create train and test splits for both entity sets
Xloc_train, Xloc_test, yloc_train, yloc_test = train_test_split(X_loc, y_loc, test_size = 0.2, random_state = 1234)
Xorg_train, Xorg_test, yorg_train, yorg_test = train_test_split(X_org, y_org, test_size = 0.2, random_state = 1234)

# Remove y rows with no data in x set
remove_yloc = [11, 13, 21, 24, 27, 39, 42, 52, 57, 58, 65, 71, 112, 116, 121, 130, 131,
            138, 139, 149, 159, 175, 188, 201, 202, 208, 209, 210, 215, 217, 239, 248, 251]

#%% Modelling for location

# Assign classfier - logistic regression
time_start = time.time()
lr = LogisticRegression(random_state=1234)
clf_loc = OneVsRestClassifier(lr)

# Fit model on training data
clf_loc.fit(Xloc_train, yloc_train)

# Make predictions on training data
yloc_pred_train  = clf_loc.predict(Xloc_train)
f1_loc = f1_score(yloc_train, yloc_pred_train, average="micro") # 8.9%
ps_loc = precision_score(yloc_train, yloc_pred_train, average="micro") # 94.28%
as_loc = accuracy_score(yloc_train, yloc_pred_train) # 4.67%%
time_stop = time.time()
training_time = time_stop-time_start
print(f"Time to train: {training_time}") # 2 seconds

# Adjust threshold as default = 0.5
yloc_pred_train  = clf_loc.predict_proba(Xloc_train)
t = 0.3 
yloc_pred_new = (yloc_pred_train >= t).astype(int)
f1_loc2 = f1_score(yloc_train, yloc_pred_new, average="micro") # 42.5%
ps_loc2 = precision_score(yloc_train, yloc_pred_new, average="micro") # 94.61%
as_loc2 = accuracy_score(yloc_train, yloc_pred_new) # 27.41%

# Adjust threshold lower again to 0.15
t = 0.15
yloc_pred_new2 = (yloc_pred_train >= t).astype(int)
f1_loc3 = f1_score(yloc_train, yloc_pred_new2, average="micro") # 59.62%
ps_loc3 = precision_score(yloc_train, yloc_pred_new2, average="micro") # 80.63%
as_loc3 = accuracy_score(yloc_train, yloc_pred_new2) # 46.45%

# Adjust threshold lower again to 0.1
t = 0.1
yloc_pred_new3 = (yloc_pred_train >= t).astype(int)
f1_loc4 = f1_score(yloc_train, yloc_pred_new3, average="micro") # 60.02%
ps_loc4 = precision_score(yloc_train, yloc_pred_new3, average="micro") # 68.49%
as_loc4 = accuracy_score(yloc_train, yloc_pred_new3) # 47.30%

# Plot F1 score comparisons
data_dict = {'50%': f1_loc, '30%': f1_loc2,
             '15%': f1_loc3, '10%': f1_loc4}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Thresholds")
plt.ylabel("F1 Score")
plt.title("F1 Score Across All Models")
plt.show()

# Plot precision score score comparisons
data_dict = {'50%': ps_loc, '30%': ps_loc2,
             '15%': ps_loc3, '10%': ps_loc4}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Thresholds")
plt.ylabel("Precision Score")
plt.title("Precision Score Across All Models")
plt.show()

# Plot accuracy score comparisons
data_dict = {'50%': as_loc, '30%': as_loc2,
             '15%': as_loc3, '10%': as_loc4}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Thresholds")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Score Across All Models")
plt.show()

# Apply model with 10% threshold to test data
yloc_pred_test  = clf_loc.predict_proba(Xloc_test)
t = 0.10
yloc_pred_test = (yloc_pred_test >= t).astype(int)
f1_loc_t = f1_score(yloc_test, yloc_pred_test, average="micro") # 47.84%
ps_loc_t = precision_score(yloc_test, yloc_pred_test, average="micro") # 58.06%
as_loc_t = accuracy_score(yloc_test, yloc_pred_test) # 35.59%

# Plot accuracy metrics
data_dict = {'AS': as_loc_t, 'PS': ps_loc_t,
             'F1': f1_loc_t}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Accuracy Metrics")
plt.ylabel("Value")
plt.title("Accuracy Metrics Across Final Model")
plt.show()

# Use case
print('The prediction vector is: ' + str(yloc_pred_test[100]))
print('This vector translates to: ' + str(multilabel_binarizer.inverse_transform(yloc_pred_test)[100]))

#%% Modelling for organisation

# Assign classfier - logistic regression
time_start = time.time()
lr = LogisticRegression(random_state=1234)
clf_org = OneVsRestClassifier(lr)

# Fit model on training data
clf_org.fit(Xorg_train, yorg_train)

# Make predictions on training data
yorg_pred_train  = clf_org.predict(Xorg_train)
f1_org = f1_score(yorg_train, yorg_pred_train, average="micro") # 29.53%
ps_org = precision_score(yorg_train, yorg_pred_train, average="micro") # 87.50%
as_org = accuracy_score(yorg_train, yorg_pred_train) # 17.76%%
time_stop = time.time()
training_time = time_stop-time_start
print(f"Time to train: {training_time}") # 1.5 seconds

# Adjust threshold as default = 0.5
yorg_pred_train  = clf_org.predict_proba(Xorg_train)
t = 0.3 
yorg_pred_new = (yorg_pred_train >= t).astype(int)
f1_org2 = f1_score(yorg_train, yorg_pred_new, average="micro") # 43.65%
ps_org2 = precision_score(yorg_train, yorg_pred_new, average="micro") # 81.50%
as_org2 = accuracy_score(yorg_train, yorg_pred_new) # 29.81%

# Adjust threshold lower again to 0.15
t = 0.15
yorg_pred_new2 = (yorg_pred_train >= t).astype(int)
f1_org3 = f1_score(yorg_train, yorg_pred_new2, average="micro") # 51.69%
ps_org3 = precision_score(yorg_train, yorg_pred_new2, average="micro") # 76.99%
as_org3 = accuracy_score(yorg_train, yorg_pred_new2) # 37.21%

# Adjust threshold lower again to 0.1
t = 0.1
yorg_pred_new3 = (yorg_pred_train >= t).astype(int)
f1_org4 = f1_score(yorg_train, yorg_pred_new3, average="micro") # 50.50%
ps_org4 = precision_score(yorg_train, yorg_pred_new3, average="micro") # 62.70%
as_org4 = accuracy_score(yorg_train, yorg_pred_new3) # 39.32%

# Plot F1 score comparisons
data_dict = {'50%': f1_org, '30%': f1_org2,
             '15%': f1_org3, '10%': f1_org4}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Thresholds")
plt.ylabel("F1 Score")
plt.title("F1 Score Across All Models")
plt.show()

# Plot precision score score comparisons
data_dict = {'50%': ps_org, '30%': ps_org2,
             '15%': ps_org3, '10%': ps_org4}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Thresholds")
plt.ylabel("Precision Score")
plt.title("Precision Score Across All Models")
plt.show()

# Plot accuracy score comparisons
data_dict = {'50%': as_org, '30%': as_org2,
             '15%': as_org3, '10%': as_org4}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Thresholds")
plt.ylabel("Accuracy Score")
plt.title("Accuracy Score Across All Models")
plt.show()

# Apply model with 10% threshold to test data
yorg_pred_test  = clf_org.predict_proba(Xorg_test)
t = 0.10
yorg_pred_test = (yorg_pred_test >= t).astype(int)
f1_org_t = f1_score(yorg_test, yorg_pred_test, average="micro") # 44.89%
ps_org_t = precision_score(yorg_test, yorg_pred_test, average="micro") # 81.5%
as_org_t = accuracy_score(yorg_test, yorg_pred_test) # 33.61%

# Plot accuracy metrics
data_dict = {'AS': as_org_t, 'PS': ps_org_t,
             'F1': f1_org_t}
courses = list(data_dict.keys())
values = list(data_dict.values())
fig = plt.figure(figsize = (10, 5))
#  Bar plot
plt.bar(courses, values, color ='brown',
        width = 0.5)
plt.xlabel("Accuracy Metrics")
plt.ylabel("Value")
plt.title("Accuracy Metrics Across Final Model")
plt.show()

# Use case
print('The prediction vector is: ' + str(yorg_pred_test[100]))
print('This vector translates to: ' + str(multilabel_binarizer2.inverse_transform(yorg_pred_test)[100]))

#%% Recommender engine

# Define function which returns closest matching tweets
def get_relevent_tweets(date, emotion, organisation, location, number):
    
    # Filter date to within range
    intel_range = intel.loc[(intel['date'] >= date-7) & (intel['date'] < date+7)]
    
    # Filter emotion inside intel_range
    intel_range = intel_range.loc[intel['emotion'] == emotion]
    
    # Perform ML to predict organisation (may need to make new clf dims)
    org_predict  = clf_org.predict_proba(intel_range['text'])
    org_predict = (org_predict >= 0.10).astype(int)
    
    # Peform ML to predict location (may need to make new clf dims)
    loc_predict  = clf_loc.predict_proba(intel_range['text'])
    loc_predict = (loc_predict >= 0.10).astype(int)
    
    # Filter results by matching organisation and location
    # reverse transformation???
    
    # Find top x (number) results
    # some kind of sort
    
    # Find actual tweet text, not stemmed version
    # Can I use filtered indices of intel_range to grab from intel_unclean?
    # push back into initial? maybe use intel_unclean which is waiting
    # use indices
    
    return similar_tweets
