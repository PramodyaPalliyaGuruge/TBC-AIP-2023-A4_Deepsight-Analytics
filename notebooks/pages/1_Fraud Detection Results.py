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
st.set_page_config(page_title="Fraud Detection Results", page_icon="📈")


# In[3]:


st.sidebar.header("Fraud Detection Results")


# In[4]:


st.markdown("# Fraud Detection Results")


# In[5]:


#test_data = r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics\data\processed\split_data\test.csv'
#df_test = pd.read_csv(test_data)

# read csv from a URL
#@st.cache_data
#def get_data() -> pd.DataFrame:
#    return pd.read_csv(test_data)

#df = get_data()


# In[6]:


# create columns for the chars
col1, col2 = st.columns(2)


# In[7]:


from io import StringIO

uploaded_file = st.file_uploader("Choose a file")
    
if uploaded_file is not None:

            df = pd.read_csv(uploaded_file)

                # Print valid and fraud transactions
            fraud=df[df.Class==1]
            valid=df[df.Class==0]
            outlier_percentage=(df.Class.value_counts()[1]/df.Class.value_counts()[0])*100
            if st.sidebar.checkbox('Show fraud and valid transaction details'):
                st.write('Fraudulent transactions are: %.3f%%'%outlier_percentage)
                st.write('Fraud Cases: ',len(fraud))
                st.write('Valid Cases: ',len(valid))


            import altair as alt

            if st.sidebar.checkbox('Plot transaction details'):
                "### Plotting transactions details"
                source = pd.DataFrame({
                    'Number of transactions': [len(fraud), len(valid)],
                    'Transaction type': ['Fraud', 'Valid']
                })

                bar_chart = alt.Chart(source).mark_bar().encode(
                    y='Number of transactions:Q',
                    x='Transaction type:O',
                )

                st.altair_chart(bar_chart, use_container_width=True)


            # Loading final model
            model_lr = pkl.load(open(r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics/models/model_lr.pkl','rb'))
            model_rf = pkl.load(open(r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics/models/model_rf.pkl','rb'))


            X_test = df.drop('Class', axis=1)
            y_test = df['Class']


            # Loading scaler
            sc = pkl.load(open(r'C:\Users\ishan\Project\TBC-AIP-2023-A4_Deepsight-Analytics/models/processers/scaler.pkl','rb'))

            X_test = sc.transform(X_test)

            X_test = pd.DataFrame(X_test)
            y_test = pd.DataFrame(y_test)

            features=X_test.columns.tolist()


            #Feature selection through feature importance
            @st.cache_data
            def feature_sort(_model,X_train,y_train):
                #feature selection
                mod=model
                # fit the model
                mod.fit(X_train, y_train)
                # get importance
                imp = mod.feature_importances_
                return imp

            start_time = timeit.default_timer()
            
            #Classifiers for feature importance
            if st.sidebar.checkbox('Top parameters'):
                "### Top parameters"
                model=model_lr
                importance = model.coef_[0]
                #importance=feature_sort(model,X_test,y_test)

                elapsed = timeit.default_timer() - start_time


                st.write('Execution Time for feature selection: %.2f minutes'%(elapsed/60))

                feature_imp=list(zip(features,importance))
                feature_sort=sorted(feature_imp, key = lambda x: x[1])

                n_top_features = st.sidebar.slider('Number of top features', min_value=5, max_value=20)

                top_features=list(list(zip(*feature_sort[-n_top_features:]))[0])

                st.write('**Top %d parameters in order of importance are**: %s'%(n_top_features,top_features[::-1]))


            #Plot of feature importance
            if st.sidebar.checkbox('Show plot of top parameters'):
                importance = model.coef_[0]
                feat_importances = pd.Series(importance)
                feat_importances.nlargest(20).plot(kind='barh',title = 'Top parameters')
                plt.xlabel('Parameter (Variable Number)')
                plt.ylabel('Importance')
                st.pyplot()

else:
            st.warning("you need to upload a csv or excel file")


# In[8]:


st.set_option('deprecation.showPyplotGlobalUse', False)


# In[ ]:




