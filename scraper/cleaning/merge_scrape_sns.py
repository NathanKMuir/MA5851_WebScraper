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

matches_sim = []
matches_ind = []
merged_df = pd.DataFrame(columns=['date', 'username', 'text', 'hashtags',
                                  'lang', 'url', 'user_desc'])
# if x% similar do this - get index, columns from sns add to scraped to get a new df

# for user in unique users
# this would filter the datasets as above to only users
content_list = test_sns['content']
for post in test_scraped['text']:
    hold_sim = []
    index_list = []
    for check in content_list:
        hold_sim.append(1-(lev(post, check)/len(post + check)))
        index_list.append(content_list[content_list == check].index[0])
    if max(hold_sim) > 0.5: 
        matches_sim.append(max(hold_sim))
        temp = list(content_list)
        temp = temp[hold_sim.index(max(hold_sim))]
        match_index = content_list[content_list == temp].index[0]
        
        # Use index to add new rows in merged dataframe
        d_row = {'date' : sns['date'][match_index], 'username' : user, 'text' : post,
                 'hashtags' : sns['hashtags'][match_index], 'lang' : sns['lang'][match_index],
                 'url' : sns['url'][match_index], 'user_desc' : sns['user_desc'][match_index]}
        merged_df = merged_df.append(d_row, ignore_index=True)
        
# The above code works - and adds to dataframe. Any issues with the below code then must be due to the way in which each user gets called or the layering of variables


            
#%%

# Get list of unique usernames and verify
scraped_users = list(set(scraped['username'])) # 8446
print(scraped['username'].nunique()) # 8446

# Initialise merged dataframe
merged_df = pd.DataFrame(columns=['date', 'username', 'text', 'hashtags',
                                  'lang', 'url', 'user_desc'])

for user in scraped_users:
    # Create dump lists on a PER USER basis
    user_sns = sns.loc[(sns['user_name'] == f'{user}')]
    user_scraped = scraped.loc[(scraped['username'] == f'{user}')]
    matches_sim = []
    matches_ind = []
    content_list = user_sns['content']
    
    # Start loop to check for similarity to scraped posts PER USER
    for post in user_scraped['text']:
        hold_sim = []
        index_list = []
        
        # TEST FROM HERE sep.
        
        # Check each post in content list against scraped post and calculate similarity
        for check in content_list:
            hold_sim.append(1-(lev(post, check)/len(post + check)))
            index_list.append(content_list[content_list == check].index[0])
        
        # After all possible matches have been checked for a scraped post, push max simalarity into new list
        if max(hold_sim) > 0.5: 
            
            # Find index in main dataframe
            matches_sim.append(max(hold_sim))
            temp = list(content_list)
            temp = temp[hold_sim.index(max(hold_sim))]
            match_index = content_list[content_list == temp].index[0]
            
            # Use index to add new rows in merged dataframe
            d_row = {'date' : sns['date'][match_index], 'username' : user, 'text' : post,
                     'hashtags' : sns['hashtags'][match_index], 'lang' : sns['lang'][match_index],
                     'url' : sns['url'][match_index], 'user_desc' : sns['user_desc'][match_index]}
            
        else:
            pass
        
 