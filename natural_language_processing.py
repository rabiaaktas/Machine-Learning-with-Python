# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 17:28:41 2020

@author: rabia
"""

#%%
import pandas as pd

#%%

data = pd.read_csv(r"gender_classifier.csv", encoding = "latin1")
data = pd.concat([data.gender, data.description], axis = 1)
data.dropna(axis = 0, inplace = True)
data.gender = [1 if each == 'female' else 0 for each in data.gender]

#%%
import re #regular expression

first_desc = data.description[4]
desc = re.sub("[^a-zA-Z]", " ", first_desc) # ^ -- not operator
desc = desc.lower() 

#%% stopwords (irrelevant words)

import nltk #natural language tool kit
nltk.download("stopwords")

# download it to the corpus file.
from nltk.corpus import stopwords

#desc = desc.split()

desc = nltk.word_tokenize(desc)

#%%
desc = [word for word in desc if not word in set(stopwords.words("english"))]
desc = "".join(desc)