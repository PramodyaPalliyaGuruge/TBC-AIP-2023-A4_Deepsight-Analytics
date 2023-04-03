#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import time # for simulating a real-time data, time loop
#import numpy as np # np mean, np random
import pandas as pd # read csv, df manipulation
import plotly.express as px # interactive charts
import matplotlib.pyplot as plt
import pickle as pkl
import timeit
import streamlit as st # data web application development


# In[2]:


#layout
st.set_page_config(page_title="Dataset", page_icon="📈")


# In[3]:


st.sidebar.header("Dataset")


# In[4]:


st.markdown("# Dataset")


# In[5]:


test_data = r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics\data\processed\split_data\test.csv'
df_test = pd.read_csv(test_data)

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(test_data)

df = get_data()


# In[6]:


# Print shape and description of the data
if st.sidebar.checkbox('Show what the dataframe looks like'):
    st.write(df.head(100))
    st.write('**Shape of the dataframe:** ',df.shape)
    st.write('### Data decription: \n',df.describe())


# In[ ]:




