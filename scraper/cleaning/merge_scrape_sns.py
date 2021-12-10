# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:58:20 2021

@author: natha
"""

# This script merges the two data sources - Selenium scraper and snscrape.

# Enable libraries
import pandas as pd
from Levenshtein import distance as lev

# Read data in from csv files
sns = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets_df.csv")
sns['date'] = pd.to_datetime(sns['date'])
scraped = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\scraped_tweets.csv")
del sns["Unnamed: 0"]
del scraped["Unnamed: 0"]

#%%

# Test table
test_sns = sns[['content', 'user_name', 'user_desc']].copy()

# Set up test user filter system
user = 'spideypooIed'
test_sns = test_sns.loc[(test_sns['user_name'] == f'{user}')]
test_scraped = scraped.loc[(scraped['username'] == f'{user}')]
                             


#%%
# Do a basic similarity check
string1 = 'party dog'
string2 = 'park mog'
lev(string1, string2)
sim_per = 1-(lev(string1, string2)/len(string1 + string2))

# if x% similar do this - get index, columns from sns add to scraped to get a new df