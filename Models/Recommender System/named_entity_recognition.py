# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 22:11:30 2021

@author: natha
"""
# This script performs named entity recognition on tweets

import pandas as pd 
import spacy
import re

# Initialise named entity recogniser
NER = spacy.load('en_core_web_sm')

# Import intel tweets dataframe
intel = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel.csv")
intel['date'] = pd.to_datetime(intel['date'])
del intel['Unnamed: 0']

# Data preparation - remove # and @ from text
intel['text'] = [re.sub(r'#', '', tweet) for tweet in intel['text']]

# Get named entities for each observation (tweet)
GPE_list = []
PERSON_list = []
ORG_list = []
for i in range(0, len(intel)):
    GPE_temp = []
    PERSON_temp = []
    ORG_temp = []
    hold = intel['text'][i]
    hold = hold.lower() # lowercase
    hold = re.sub(r'[^\w\s]', '', hold) # remove punctuation
    hold = re.sub(r'[0-9]', '', hold) # remove numbers
    review = NER(hold)
    for word in review.ents:
        if word.label_ == "PERSON":
            PERSON_temp.append(word.text)
        if word.label_ == "GPE":
            GPE_temp.append(word.text)
        if word.label_ == "ORG":
            ORG_temp.append(word.text)
    GPE_list.append(GPE_temp)
    PERSON_list.append(PERSON_temp)
    ORG_list.append(ORG_temp)

# Add named entities onto intel dataframe
intel['location'] = GPE_list
intel['person'] = PERSON_list
intel['organisation'] = ORG_list

# Export intel to csv
intel.to_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel_NER.csv")
