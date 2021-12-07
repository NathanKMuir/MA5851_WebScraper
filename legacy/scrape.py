# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 17:41:19 2021

@author: natha
"""

# Import packages
import pandas as pd
import os
import snscrape.modules.twitter as sntwitter

# Set working directory
wd = "C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3"
os.chdir(wd)

print("...script successfully initialised.")

# Creating list to append tweet data
tweets_list = []

# Use TwitterSearchScraper to append tweets to list
for i, tweet in enumerate(sntwitter.TwitterSearchScraper(''''improvised explosive device
                                                         OR ied
                                                         OR improvised explosive
                                                         OR bomb
                                                         OR explosive device
                                                         OR vbied
                                                         OR suicide bomber
                                                         OR suicide bomb
                                                         OR car bomb
                                                         OR pbied
                                                         OR uas
                                                         OR uav
                                                         OR suas
                                                         OR loitering munition
                                                         OR unmanned aerial system
                                                         OR unmanned aerial vehicle
                                                         OR drone attack
                                                         OR drone explosive
                                                         since:2020-12-01 until:2021-12-01''').get_items()): # declare username
    if i > 499999:
        break
    tweets_list.append(tweet)
tweets_df = pd.DataFrame(tweets_list)    
    
# Grab all user data
hold = (tweets_df['user'])
hold = hold.to_dict()

# For group
user_name = []
user_id = []
user_desc = []
user_followers = []
user_location = []
user_created = []

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

hold = tweets_df[['date', 'url', 'content', 'hashtags', 'id', 'lang', 'likeCount', 'replyCount',
                  'retweetCount', 'quoteCount', 'outlinks', 'media', 'coordinates', 'place']]

tweets_df = pd.merge(hold, user_data, how='inner', right_index=True, left_index=True)
