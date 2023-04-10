#!/usr/bin/env python
# coding: utf-8

# In[5]:


#import time # for simulating a real-time data, time loop
#import numpy as np # np mean, np random
import pandas as pd # read csv, df manipulation
#import plotly.express as px # interactive charts
#import matplotlib.pyplot as plt
#import pickle as pkl
import timeit
import streamlit as st # data web application development


# In[6]:


#layout
st.set_page_config(
    page_title="Credit Card Fraud Detection System",
    page_icon="💳",
    layout="wide",
)


# In[7]:


# dashboard title
#st.image("credit_card.png", width=100)
htp="https://raw.githubusercontent.com/PramodyaPalliyaGuruge/TBC-AIP-2023-A4_Deepsight-Analytics/main/notebooks/credit_card.svg.png"
st.image(htp, width=100)
st.title("Welcome to Deepsight Credit Card Fraud Detection System")


# In[8]:


st.markdown(
    """
    ## Tired of dealing with credit card fraud in your business?
    
    We go beyond simple credit card fraud detection. We generate trust between your company 
    and your good customers, to improve the total results of your business.
    
    Get the highest accuracy rates and lowest incorrect fruad detection rates in the industry with an innovative 
    fraud prevention solution that’s always customized to your business.
    **👈 Navigate the tabs from the sidebar** to upload and see results
    
    ### Summary of each tab
    - **Fraud Detection Result**: Upload Data, Transaction Details and Top parameters
    - **Data Science (Statistics)**: Model Selection, Confusion Matrix and Classification Report
    - **Dataset**: View Data and Summary Statistics
       
"""
)

# In[ ]:




