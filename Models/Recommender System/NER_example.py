# -*- coding: utf-8 -*-
"""
Created on Sat Dec 11 23:27:11 2021

@author: natha
"""

# NER example

import pandas as pd 
import spacy
from spacy import displacy

# Initialise named entity recogniser
NER = spacy.load('en_core_web_sm')

# Import uncleaned tweets dataframe and add column
intel = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel.csv")
intel['date'] = pd.to_datetime(intel['date'])
del intel['Unnamed: 0']

# Generate test string
test = "Irony as IED placed by Al Shabaab kills 15 of its own militants in Kenya"

test1 = NER(test)
for word in test1.ents:
    print(word.text, word.label_)