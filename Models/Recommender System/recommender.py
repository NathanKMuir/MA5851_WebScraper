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

# Import intel tweets dataframe
intel = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel_NER.csv")
intel['date'] = pd.to_datetime(intel['date'])
del intel['Unnamed: 0']

#%% Visualisations of entities 

# Visualise location
locations = intel['location']
for i in (range(0, len(locations))):
    locations[i] = re.sub(r'[^\w\s]', '', locations[i])
word_list = []
for i in range(0, len(locations)):
    review = locations[i].split()
    for word in review:
        word_list.append(word)
all_locations = nltk.FreqDist(word_list)
unique_loc = len(all_locations)
print(f'The number of unique locations is: {unique_loc}') # 572
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
print(f'The number of unique people is: {unique_people}') # 1093
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
print(f'The number of unique people is: {unique_orgs}') # 1510
all_orgs = pd.DataFrame({'organisation':list(all_orgs.keys()),
                             'count':list(all_orgs.values())})
g_org = all_orgs.nlargest(columns="count", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g_org, x= "count", y = "organisation") 
ax.set(ylabel = 'count') 
plt.show()

#%% Pre-processing to create corpus from intel text

