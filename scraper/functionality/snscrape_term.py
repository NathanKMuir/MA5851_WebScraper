# -*- coding: utf-8 -*-
"""
Created on Wed Dec  8 17:05:52 2021

@author: natha
"""
# This script returns a series of tweets based on search term.

import pandas as pd
import snscrape.modules.twitter as sntwitter

# Use TwitterSearchScraper to append tweets to list
def search_twitter(search_terms, since, until):
    tweets_list = []
    for search in search_terms:
        for i, tweet in enumerate(sntwitter.TwitterSearchScraper(f'{search} since:{since} until:{until}').get_items()): # declare username
            tweets_list.append(tweet)
    # Store all returned tweets in a dataframe
    tweets_df = pd.DataFrame(tweets_list)    
    # Alert notification & return dataframe
    print("...tweets scraped.")
    return(tweets_df)

# Creating list and search parameters
search_terms = ("improvised explosive device", "IED", "improvised explosive", "explosive device", "suicide bomb",
                "suicide bomber", "car bomb", "vbied", "pbied", "voied", "uas", "suas", "uav", "loitering munition",
                "unmanned aerial system", "unmanned aerial vehicle", "drone attack", "drone explosive")
since = "2021-09-01"
until = "2021-12-01"
   
# Call function
tweets_df = search_twitter(search_terms, since, until)

#%% Get fields and push data into dataframe

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