#!/usr/bin/env python
# coding: utf-8

# ## Import packages

# In[25]:


import numpy as np
import pandas as pd
import seaborn as sns
import scipy
import skimpy
from skimpy import skim
from scipy import stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.stats.multicomp import MultiComparison
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.proportion import proportions_chisquare


# ## Import data

# In[16]:


Meteor = pd.read_csv('C:/Users/steph/Documents/GitHub/Final-Project/Meteorite_Landings/Meteorite_Landings.csv')


# In[18]:


Meteor.info()


# In[19]:


Meteor.describe()


# In[27]:


Meteor2= Meteor.dropna()


# In[32]:


Meteor.columns


# In[34]:


Meteor.head()


# In[35]:


Meteor2.head()


# In[36]:


Meteor2[['longitude', 'latitude']] = Meteor2['GeoLocation'].str.split(' ', 1, expand=True)


# In[37]:


Meteor2.head()


# In[38]:


Meteor2.dropna(inplace = True)


# In[39]:


Meteor2.head()


# In[40]:


Meteor2.columns


# In[43]:


Meteor3 = Meteor2.drop("GeoLocation", axis = 1)


# In[44]:


Meteor3.head()


# In[74]:


Meteor3.fall.value_counts()


# In[48]:


Meteor3.recclass.value_counts()


# In[71]:


Meteor3.nametype.value_counts()


# In[72]:


Meteor3.id.value_counts()


# In[75]:


cleanup = {"nametype": {"Valid": 0, "Relict": 1},
          "recclass": {"L6": 0, "L5": 1, "H4": 2, "H5": 3, "H6": 4},
           "fall": {"Found": 1, "Fell": 2}}
Meteor3.replace(cleanup, inplace=True)


# In[83]:


Meteor3.head()


# In[76]:


X = Meteor3[['recclass', 'mass (g)','reclat', 'reclong', 'id', 'nametype', 'fall' ]]


# In[77]:


y = Meteor3[['longitude', 'latitude']]


# In[78]:


from sklearn.model_selection import train_test_split


# In[79]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[80]:


from sklearn.ensemble import RandomForestRegressor


# In[81]:


rfr = RandomForestRegressor()

rfr.fit(X_train, y_train)
# In[82]:


y_pred = rfr.predict(X_test)


# In[ ]:




