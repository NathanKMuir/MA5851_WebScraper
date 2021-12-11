# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 11:47:53 2021

@author: natha
"""

# This script builds a emotional tone predictive model using the GoEmotions dataset.

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Load in each segment of the GoEmotions data
df1 = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\goemotions_1.csv")
df2 = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\goemotions_2.csv")
df3 = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\goemotions_3.csv")

# Concatenate data
frames = [df1, df2, df3]
GoEmotions = pd.concat(frames)
print(GoEmotions.shape) # 211225, 37

# Filter out unclear examples
GoEmotions = GoEmotions.loc[(GoEmotions['example_very_unclear'] == False)]
print(GoEmotions.shape) # 207814, 37
GoEmotions = GoEmotions.reset_index(drop=True)

# Remove unneccessary columns
del GoEmotions['id']
del GoEmotions['author']
del GoEmotions['subreddit']
del GoEmotions['link_id']
del GoEmotions['parent_id']
del GoEmotions['created_utc']
del GoEmotions['rater_id']
del GoEmotions['example_very_unclear']

# Get name of emotions and sum of each
emotion_names = list(GoEmotions.columns)
emotion_names = emotion_names[1:]
emotion_sum = []
for emotion in emotion_names:
    total = GoEmotions[f'{emotion}'].sum()
    emotion_sum.append(total)
emotion_sum = pd.DataFrame(list(zip(emotion_names, emotion_sum)),
                           columns = ['Emotion', 'Count'])    

# Plot frequency distribution of emotions
g = emotion_sum.nlargest(columns="Count", n = 28) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Emotion") 
ax.set(ylabel = 'count') 
plt.show()

# Make all text lowercase
GoEmotions['text'] = [word.lower() for word in GoEmotions['text']]

# Remove all punctuation from text
for i in (range(0, len(GoEmotions))):
    GoEmotions['text'][i] = re.sub(r'[^\w\s]', '', GoEmotions['text'][i])
print('Punctuation removed.')
    
# Remove numbers
for i in (range(0, len(GoEmotions))):
    GoEmotions['text'][i] = re.sub(r'[0-9]', '', GoEmotions['text'][i])
print('Numbers removed.')

# Load and configure stopwords
all_stopwords = stopwords.words('english')
additional_stopwords = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'j', 'k', 'l', 'm',
                        'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
for letter in additional_stopwords:
    all_stopwords.append(letter)

# Remove stopwords
passwords = []
for sentence in range(0, len(GoEmotions['text'])):
    review = GoEmotions['text'][sentence].split()
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
        word_list.append(word) # 1,426,741 total words

# Generate frequency dictionary
freqDict =  dict()
visited = set()
for element in word_list:
    if element in visited:
        freqDict[element] = freqDict[element] + 1
    else:
        freqDict[element] = 1
        visited.add(element)
print('The number of unique words in the corpus is: ' + str(len(freqDict))) # 22556
word_freq = pd.Series(freqDict).sort_values(ascending=False)

# Visualise top 50 words
word_list1 = nltk.FreqDist(word_list)
word_list1 = pd.DataFrame({'Word':list(word_list1.keys()),
                             'Count':list(word_list1.values())})
g = word_list1.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Word") 
ax.set(ylabel = 'Count') 
plt.show()

# Push corpus back into GoEmotions dataframe & export to CSV
GoEmotions['text'] = corpus
GoEmotions.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\GoEmotions.csv")
