#!/usr/bin/env python
# coding: utf-8

# # Data Splitting

# In[2]:


# Installing imblearn library
get_ipython().system('pip install imblearn')


# In[22]:


# Importing libraries

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import pickle as pkl
import warnings
warnings.filterwarnings('ignore') 


# In[23]:


# loading the dataset
raw_data = r'..\data\raw\Deep Sight Analytics creditcard_cc.csv'
df=pd.read_csv(raw_data)


# In[24]:


df.head(5)


# In[25]:


# Checking size of creditcard Dataset
df.shape


# In[26]:


df['Class'].value_counts()


# In[27]:


sns.countplot(df['Class'])
plt.title('Distribution of the Class')


# In[28]:


def split_dataset(input_filepath, output_folder, test_size):
    '''This function splits data and saves train and test files
    Parameters
    ---------
    features : pandas dataframe
        Input data to be split

    Returns
    -------
    None
    '''
    df =pd.read_csv(input_filepath)
    train, test = train_test_split(df, test_size = test_size, random_state = 0)
    
    train.to_csv(output_folder+r'\train.csv', index=False)
    test.to_csv(output_folder+r'\test.csv', index=False)


# In[29]:


split_dataset(r'..\data\raw\Deep Sight Analytics creditcard_cc.csv', r'..\data\processed\split_data', 0.25)


# In[30]:


file_path = '../models/processers/split_data.pkl'
pkl.dump(split_dataset, open(file_path, 'wb'))


# In[ ]:




