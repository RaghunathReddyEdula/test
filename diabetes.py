#!/usr/bin/env python
# coding: utf-8

# In[5]:

print('Importing Dependencies.....')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


# In[2]:

print('Reading the data......')
dataset = pd.read_csv('diabetes.csv')


# In[3]:


X = dataset.iloc[:, :3]
y = dataset.iloc[:, -1]


# In[6]:

print('Building Model...........')
regressor = LogisticRegression()


# In[7]:


regressor.fit(X, y)


# In[8]:


print('saving model as pkl file.......')
pickle.dump(regressor, open('model.pkl','wb'))


# In[9]:


model = pickle.load(open('model.pkl','rb'))


# In[10]:


# print(model.predict([[1, 1, 2]]))


# In[ ]:




