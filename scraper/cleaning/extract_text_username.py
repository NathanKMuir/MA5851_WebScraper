# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:03:16 2021

@author: natha
"""

import pandas as pd
import re
import numpy as np

# Import scraped tweets
content = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\content_df.csv")
content = list(content['0'])

# Remove duplicated entries at back of list
del content[7341:7928]

# Extract usernames
usernames = []
for i in content:
    usernames.append(re.findall("@([/(?<!\w)@(\w+)/]+)", i, re.I))

# Extract text belonging to user
content_list = []
# Now add this onto full loop
for i in range(0, len(content)):
    screen = content[i]
    users = usernames[i]
    for user in users:
        if users.index(user) < (len(users)-1):
            content_add = screen[screen.find(user)+(len(user)) : screen.find(users[users.index(user)+1])-1]
            content_list.append(content_add)
            screen = screen.replace("@" + user, "", 1)   
        else:
            content_add = screen[screen.find(user)+(len(user)) : screen.find("New to Twitter?")]
            content_list.append(content_add)

# Create list of users as they appear in the text
user_list = []
for i in usernames:
    for user in i:
        user_list.append(user)
        
# Join users and their tweets together
scraped_tweets = pd.DataFrame({"username" : user_list, "text" : content_list})

# Remove date (inconsistent)
for i in range(0, len(scraped_tweets)-1):
    try:
        if "·" in scraped_tweets['text'][i]:
            hold = scraped_tweets['text'][i]
            hold = hold.split()
            del hold[0]
            hold[0] = re.sub(r'[0-9]', '', hold[0])
            hold = " ".join(hold)
            scraped_tweets['text'][i] = hold
    except:
        pass
    
# Filter out empty tweets
scraped_tweets.shape # 121203, 2
scraped_tweets['text'] = scraped_tweets['text'].replace(r'^\s*$', np.NaN, regex=True)
scraped_tweets.dropna(subset = ['text'], inplace = True) 
scraped_tweets.shape # 101391, 2
scraped_tweets = scraped_tweets.reset_index(drop=True)

# Filter out parsing errors where characters are over tweet limit
scraped_tweets['char_count'] = 1
for i in range(0, len(scraped_tweets)):
    scraped_tweets['char_count'][i] = len(scraped_tweets['text'][i])
scraped_tweets = scraped_tweets.loc[(scraped_tweets['char_count'] >= 0)
                                          & (scraped_tweets['char_count'] < 240)]
del scraped_tweets['char_count']
scraped_tweets.shape # 69468, 2

# Delete last term in each string (screen name of next user)
for i in range(0, len(scraped_tweets)-1):
    try:
        hold = scraped_tweets['text'][i]
        hold = hold.split()
        del hold[-1]
        hold = " ".join(hold)
        scraped_tweets['text'][i] = hold
    except:
        pass
    
# Remove new blank elements if tweet consisted of only one term
scraped_tweets['text'] = scraped_tweets['text'].replace(r'^\s*$', np.NaN, regex=True)
scraped_tweets.dropna(subset = ['text'], inplace = True) 
scraped_tweets.shape # 65005, 2
scraped_tweets = scraped_tweets.reset_index(drop=True)

# Print example
print(scraped_tweets['username'][388] + ": " + scraped_tweets['text'][388])
# Rita_Katz: AQAP claims killing two Houthis in an IED attack on a motorcycle in Yemen's Abyan governorate,
# marking the group's first claimed attack there since March 2021 (eight months ago).Quote TweetSITE Intel - Jihadist

# Remove duplicates
scraped_tweets = scraped_tweets.drop_duplicates(keep='first')
scraped_tweets.shape # 19088, 2

# Export as CSV
scraped_tweets.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\scraped_tweets.csv")
    