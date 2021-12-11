# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 10:43:16 2021

@author: natha
"""

# This script takes the merged data and cleans it, ready for analysis and modelling.

import pandas as pd
import nltk
import matplotlib.pyplot as plt
import seaborn as sns
import re
from statistics import mean

# Import data and resetto previous state
merged_df = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\merged_df.csv")
merged_df['date'] = pd.to_datetime(merged_df['date'])
del merged_df['Unnamed: 0']

# Filter for tweets which contain the search term in the text
print(len(merged_df)) # 13102
key_words = []
for tweet in merged_df['text']:
    review = []
    words = tweet.split()
    words = [word.lower() for word in words]
    for word in words:
        if word == "ied":
            review.append(word)
    key_words.append(review)

# Add new column with class labels
merged_df['search_term'] = key_words

# Remove rows with no search terms
hold = []
for word in key_words:
    h = " ".join(map(str,word))
    hold.append(h)
merged_df['temp'] = hold
tweets = merged_df.loc[(merged_df['temp'] != "")]
del tweets['temp']
del tweets['search_term']
print(len(tweets)) # 5510 - so, 7952 tweets contained no mention of the search term

# Visualise languages
all_lan = nltk.FreqDist(tweets['lang'])
all_lan = pd.DataFrame({'Language':list(all_lan.keys()),
                             'Count':list(all_lan.values())})
g = all_lan.nlargest(columns="Count", n = 20) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Language") 
ax.set(ylabel = 'Count') 
plt.show()

# Filter for tweets in English
tweets = tweets.loc[tweets['lang'] == 'en']
print(len(tweets)) # 3328
del tweets['lang']
tweets = tweets.reset_index(drop=True)

# Remove URLs from text and user_description
for i in (range(0, len(tweets))):
    tweets['text'][i] = re.sub(r'http\S+', '', tweets['text'][i])
tweets['user_desc'].fillna("", inplace=True) # set empty desc to blank spaces
for i in (range(0, len(tweets))):
    tweets['user_desc'][i] = re.sub(r'http\S+', '', tweets['user_desc'][i])

# Remove emojis from text, username and user_description
emoji_pattern = re.compile("["
         u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                           "]+", flags=re.UNICODE)
for i in (range(0, len(tweets))):
    tweets['text'][i] = re.sub(emoji_pattern, '', tweets['text'][i])
for i in (range(0, len(tweets))):
    tweets['user_desc'][i] = re.sub(emoji_pattern, '', tweets['user_desc'][i])
for i in (range(0, len(tweets))):
    tweets['username'][i] = re.sub(emoji_pattern, '', tweets['username'][i])
    
# Get number of unique users & mean posts
print(tweets['username'].nunique()) #1650
all_users = nltk.FreqDist(tweets['username'])
all_users = pd.DataFrame({'Username':list(all_users.keys()),
                             'Posts':list(all_users.values())})
print(mean(all_users['Posts'])) # approx 2.02

# Get plot of top 30 users
g = all_users.nlargest(columns="Posts", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Posts", y = "Username") 
ax.set(ylabel = 'Posts') 
plt.show()

# Plot top 30 hashtags
tweets['hashtags'].fillna("", inplace=True) # set empty hashtags to blank spaces
hashtags = list(tweets['hashtags'])
for i in (range(0, len(hashtags))):
    hashtags[i] = re.sub(r'[^\w\s]', '', hashtags[i])
word_list = []
for i in range(0, len(hashtags)):
    review = hashtags[i].split()
    for word in review:
        word_list.append(word)
all_hash = nltk.FreqDist(word_list)
all_hash = pd.DataFrame({'hashtags':list(all_hash.keys()),
                             'count':list(all_hash.values())})
g = all_hash.nlargest(columns="count", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "count", y = "hashtags") 
ax.set(ylabel = 'count') 
plt.show()

# Export final scrape to CSV
tweets.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets.csv")