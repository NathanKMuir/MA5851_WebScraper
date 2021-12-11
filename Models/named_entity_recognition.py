# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 22:11:30 2021

@author: natha
"""
# This script performs named entity recognition on tweets

# Import uncleaned tweets dataframe and add column
tweets = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\tweets.csv")
tweets['date'] = pd.to_datetime(tweets['date'])
del tweets['Unnamed: 0']
tweets['emotion'] = emotions['column']