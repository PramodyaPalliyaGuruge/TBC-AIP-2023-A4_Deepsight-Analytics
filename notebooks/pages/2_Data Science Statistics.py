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
st.set_page_config(page_title="Data Science Statistics", page_icon="📈")


# In[3]:


st.sidebar.header("Data Science Statistics")


# In[4]:


st.markdown("# Model Selection and Prediction")


# In[5]:


test_data = r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics\data\processed\split_data\test.csv'
df_test = pd.read_csv(test_data)

# read csv from a URL
@st.cache_data
def get_data() -> pd.DataFrame:
    return pd.read_csv(test_data)

df = get_data()


# In[8]:


# Loading final model
model_lr = pkl.load(open(r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics/models/model_lr.pkl','rb'))
model_rf = pkl.load(open(r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics/models/model_rf.pkl','rb'))


# In[9]:


# Loading scaler
sc = pkl.load(open(r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics/models/processers/scaler.pkl','rb'))


# In[10]:


X_test = df.drop('Class', axis=1)
y_test = df['Class']


# In[11]:


X_test = sc.transform(X_test)


# In[12]:


X_test = pd.DataFrame(X_test)
y_test = pd.DataFrame(y_test)


# In[13]:


from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report
from yellowbrick.classifier import ClassificationReport


# In[15]:


def compute_performance(model,X_test,y_test):
    start_time = timeit.default_timer()
    y_pred = model.predict(X_test)
    rec = round(recall_score(y_test,y_pred)*100, 2)
    st.write('**Recall score (%):** ',rec)
    
    

    "### Confusion Matrix"
    cm = confusion_matrix(y_test, y_pred, labels = [0, 1])
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    st.pyplot()
    
    
    "### Summarized Classification Report"
    visualizer = ClassificationReport(model, support=True)
    visualizer.score(X_test, y_test)  # Evaluate the model on the test data
    visualizer.show()
    st.pyplot()
    #cr=classification_report(y_test, y_pred)
    #'Classification Report: ',cr
    
    elapsed = timeit.default_timer() - start_time
    'Execution Time for performance computation: %.2f minutes'%(elapsed/60)

# In[16]:


#Run different classification models
if st.sidebar.checkbox('Run a credit card fraud detection model'):
    
    alg=['Logistic Regression (Recommended)','Random Forest']
    classifier = st.sidebar.selectbox('Which algorithm?', alg)
    
    if classifier=='Logistic Regression (Recommended)':
        model=model_lr
        compute_performance(model,X_test,y_test)
        
    elif classifier == 'Random Forest':
        model=model_rf
        compute_performance(model,X_test,y_test)


# In[17]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[ ]:




