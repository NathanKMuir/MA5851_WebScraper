# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 14:03:16 2021

@author: natha
"""

import pandas as pd
import re

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
for k in range(0, len(content)):
    users = usernames[k]
    screen = content[k]
    for i in range(0, len(users)):
        if i > len(users):
            content_add = screen[screen.find(users[i])+(len(users[i])) : screen.find(users[i+1])]
            content_list.append(content_add)
        else:
            content_add = screen[screen.find(users[i])+(len(users[i])) : screen.find("New to Twitter?")]
            content_list.append(content_add)

# Create list of users as they appear in the text
user_list = []
for i in usernames:
    for user in i:
        user_list.append(user)


#%% testing


         
         

