# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 14:58:20 2021

@author: natha
"""

# Enable libraries
import pandas as pd
from Levenshtein import distance as lev

# Read data in from csv files
sns = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets_df.csv")
sns['date'] = pd.to_datetime(sns['date'])
scraped = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\scraped_tweets.csv")
del sns["Unnamed: 0"]
del scraped["Unnamed: 0"]

# Get list of unique usernames and verify
scraped_users = list(set(scraped['username'])) # 8446
scraped_users = scraped_users # reset index
sns_users = list(set(sns['user_name'])) # 96519
print(scraped['username'].nunique()) # 8446
print(sns['user_name'].nunique()) # 96519

# Initialise merged dataframe
merged_df = pd.DataFrame(columns=['date', 'username', 'text', 'hashtags',
                                  'lang', 'url', 'user_desc'])
i_control = 0

for user in scraped_users:
    i_control += 1
    
    if user in sns_users:
        test_sns = sns.loc[(sns['user_name'] == f'{user}')]
        test_scraped = scraped.loc[(scraped['username'] == f'{user}')]
        content_list = test_sns['content']
    
        for post in test_scraped['text']:
            hold_sim = []
            index_list = []
            for check in content_list:
                hold_sim.append(1-(lev(post, check)/len(post + check)))
                index_list.append(content_list[content_list == check].index[0])
            if max(hold_sim) > 0.5: 
                temp = list(content_list)
                temp = temp[hold_sim.index(max(hold_sim))]
                match_index = content_list[content_list == temp].index[0]
                
                # Use index to add new rows in merged dataframe
                d_row = {'date' : sns['date'][match_index], 'username' : user, 'text' : post,
                         'hashtags' : sns['hashtags'][match_index], 'lang' : sns['lang'][match_index],
                         'url' : sns['url'][match_index], 'user_desc' : sns['user_desc'][match_index]}
                merged_df = merged_df.append(d_row, ignore_index=True)
        
    # Iteration marker
    print(f"Users processed: {i_control} of 8446")

# Export to CSV
merged_df.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\merged_df.csv")
