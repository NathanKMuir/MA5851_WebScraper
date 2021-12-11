# -*- coding: utf-8 -*-
"""
Created on Sun Dec 12 00:50:56 2021

@author: natha
"""

# This script creates machine learning algorithms for the prediction of entities, and then creates a recommender system.

import pandas as pd

# Import intel tweets dataframe
intel = pd.read_csv("C:\\Users\\natha\\Documents\\Master of Data Science\\MA5851 Natural Language Processing\\A3\\intel_NER.csv")
intel['date'] = pd.to_datetime(intel['date'])
del intel['Unnamed: 0']

# Visualise location
