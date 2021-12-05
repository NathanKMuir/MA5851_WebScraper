# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:41:19 2021

@author: natha
"""
#%% DOCUMENT TWO
#
#
#
#

# Import packages
import pandas as pd
import os
import snscrape.modules.twitter as sntwitter
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import re
from statistics import mean

# Set working directory
wd = "C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3"
os.chdir(wd)

# Creating list and search parameters
tweets_list = []
search_terms = ("improvised explosive device", "IED", "improvised explosive", "explosive device", "suicide bomb",
                "suicide bomber", "car bomb", "vbied", "pbied", "voied", "uas", "suas", "uav", "loitering munition",
                "unmanned aerial system", "unmanned aerial vehicle", "drone attack", "drone explosive")
since = "2021-09-01"
until = "2021-12-01"

# Use TwitterSearchScraper to append tweets to list
for search in search_terms:
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{search} since:{since} until:{until}').get_items()): # declare username
        tweets_list.append(tweet)
        
# Store all returned tweets in a dataframe
tweets_df = pd.DataFrame(tweets_list)    

# Alert notification
print("...tweets scraped.")

#%% Push data into dataframe, limit to month of November

# Grab all user data
hold = (tweets_df['user'])
hold = hold.to_dict()

# Initialise holding columns for each level of user data
user_name = []
user_id = []
user_desc = []
user_followers = []
user_location = []
user_created = []

# Grab user data and push into one data frame
for i in hold:
    user_name.append(hold[i]['username'])
    user_id.append(hold[i]['id'])
    user_desc.append(hold[i]['description'])
    user_followers.append(hold[i]['followersCount'])
    user_location.append(hold[i]['location'])
    user_created.append(hold[i]['created'])
zipped = list(zip(user_name, user_id, user_desc, user_location, user_created, user_followers))
user_data = pd.DataFrame(zipped, columns=['user_name', 'user_id', 'user_desc', 
                                          'user_location', 'user_created', 'user_followers'])

# Select the desired fields from returned tweets
hold = tweets_df[['date', 'url', 'content', 'hashtags', 'id', 'lang', 'likeCount', 'replyCount',
                  'retweetCount', 'quoteCount', 'outlinks', 'media', 'coordinates', 'place']]

# Join user data and selected fields in DF and export to CSV in local directory
tweets_df = pd.merge(hold, user_data, how='inner', right_index=True, left_index=True)
tweets_df.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets_df.csv")
tweets_df = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets_df.csv")
tweets_df['date'] = pd.to_datetime(tweets_df['date'])

# Filter for tweets occuring in November only, and export to CSV
tweets = tweets_df.loc[(tweets_df['date'] >= '2021-11-01')
                                          & (tweets_df['date'] < '2021-12-01')]
tweets.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets.csv")

#%% Cleaning

# Reset index
tweets = tweets.reset_index(drop=True)

# Remove NA values and replace with blanks
tweets = tweets.fillna('')

# Remove unnamed column
del tweets['Unnamed: 0']

# Get total length
print(len(tweets)) # 199566

# Remove duplicates
tweets = tweets.drop_duplicates(['id'], keep='last')
print(len(tweets)) #198546

# Verify by checking number of unique Tweet IDs
print(tweets['id'].nunique()) #198546

# Visualise representation of languages
all_lan = nltk.FreqDist(tweets['lang'])
all_lan = pd.DataFrame({'Language':list(all_lan.keys()),
                             'Count':list(all_lan.values())})
g = all_lan.nlargest(columns="Count", n = 50) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Count", y = "Language") 
ax.set(ylabel = 'Count') 
plt.show()

# Filter for tweets in English
tweets = tweets.loc[tweets['lang'] == 'en']
print(len(tweets)) # 42008
del tweets['lang']

# Reset index
tweets = tweets.reset_index(drop=True)

# Create new column 'location', providing hierarchy to returned geographical fields
tweets['location'] = ' '
for i in (range(0, len(tweets))):
    if tweets['place'][i] == '':
        if tweets['coordinates'][i] == '':
            if tweets['user_location'][i] == '':
                tweets['location'][i] = ''
            else:
                tweets['location'][i] = tweets['user_location'][i]
        else:
            tweets['location'][i] = tweets['coordinates'][i]
    else:
        tweets['location'][i] = tweets['place'][i]

# Visualise frequency of geographic fields having no value and benefit of general 'location'
hold = {'place':(tweets.loc[tweets['place'] == ''].count().iloc[0]),
        'coordinates':(tweets.loc[tweets['coordinates'] == ''].count().iloc[0]),
        'user_location':(tweets.loc[tweets['user_location'] == ''].count().iloc[0]),
        'location':(tweets.loc[tweets['location'] == ''].count().iloc[0])}
names = list(hold.keys())      
values = list(hold.values())
plt.bar(range(len(hold)), values, tick_label=names)
plt.show()

# Remove extraneous geographic fields
del tweets['place']
del tweets['coordinates']
del tweets['user_location']

# Get dimensions
print(tweets.shape) # 42008, 17

# Get number of unique users & mean posts
print(tweets['user_name'].nunique()) #18109
all_users = nltk.FreqDist(tweets['user_name'])
all_users = pd.DataFrame({'Username':list(all_users.keys()),
                             'Posts':list(all_users.values())})
print(mean(all_users.values())) # approx 2.32

# Get plot of top 30 users
g = all_users.nlargest(columns="Posts", n = 30) 
plt.figure(figsize=(12,15)) 
ax = sns.barplot(data=g, x= "Posts", y = "Username") 
ax.set(ylabel = 'Posts') 
plt.show()

# Plot top 30 hashtags
hashtags = tweets['hashtags']
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

# Plot representation of location?

# Clean/standardise location?

# Plot number of tweets corresponding to each search term


#%% DOCUMENT THREE
#
#
#
#

# Remove URLs from content and user description
for i in (range(0, len(tweets))):
    tweets['content'][i] = re.sub(r'http\S+', '', tweets['content'][i])
for i in (range(0, len(tweets))):
    tweets['user_desc'][i] = re.sub(r'http\S+', '', tweets['user_desc'][i])

