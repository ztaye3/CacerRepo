#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Python for Data Science, Homework, template: </h4> </center>
# <center> <h1 style="color:#303030">Breast Cancer Selection</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong> Practice binary classification on Breast Cancer data</p>
#         <strong> Sections:</strong>
#         <a id="P0" name="P0"></a>
#         <ol>
#             <li> <a style="color:#303030" href="#SU">Set Up </a> </li>
#             <li> <a style="color:#303030" href="#P1">Exploratory Data Analysis</a></li>
#             <li> <a style="color:#303030" href="#P2">Modeling</a></li>
#         </ol>
#         <strong>Topics Trained:</strong> Binary Classification.
#     </div>
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/" title="momentum"> SIT Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day1/index.html" title="momentum">Week 2 Day 1, Applied Machine Learning</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/17X_OTM8Zqg-r4XEakCxwU6VN1OsJpHh7?usp=sharing" title="momentum"> Assignment, Classification of breast cancer cells</a>
# </strong></nav>

# **Team Members:** 
# 
# *   Zekarias Taye Hirpo
# *   Agbeyeye Koffi Ledi
# *   Mohit Kumar Bassak
# 
# 
# 
# 

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# **Package install**

# In[45]:


##get_ipython().system(u'sudo apt-get install build-essential swig')
##get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
##get_ipython().system(u'pip install auto-sklearn ')


# In[46]:


##get_ipython().system(u'pip install pipelineprofiler')


# In[47]:


##get_ipython().system(u'pip install shap')


# In[48]:


##get_ipython().system(u'pip install --upgrade plotly')


# In[49]:


##get_ipython().system(u'pip3 install -U scikit-learn')


# In[50]:


import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly
plotly.__version__

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots


# In[51]:


from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing
import datetime



# In[52]:


import autosklearn.classification
import PipelineProfiler


# In[53]:


logging.basicConfig(filename = 'logs.log' , level = logging.INFO)


# In[75]:


import shap
from sklearn.metrics import mean_squared_error
from joblib import dump


# Connect to your Google Drive

# In[55]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# Options and settings

# In[56]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/data/CancerRepo/data/raw/"


# In[57]:


model_path = "/content/drive/MyDrive/Introduction2DataScience/data/CancerRepo/models/"


# In[58]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[59]:


logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# **Observation:** Those are the necessary packages and resources we should import before proceeding to data anlysis.

# Please Download the data from [this source](https://drive.google.com/file/d/1af2YyHIp__OdpuUeOZFwmwOvCsS0Arla/view?usp=sharing), and upload it on your introduction2DS/data google drive folder.

# # Loading Data and Train-Test Split

# In[60]:


dataset = pd.read_csv(f'{data_path}data-breast-cancer.csv')


# In[61]:


drop_column = ['id', 'Unnamed: 32']
dataset.drop(drop_column,axis=1, inplace=True)


# In[62]:


test_size = 0.2
random_state = 0


# In[63]:


train, test = train_test_split(dataset, test_size=test_size, random_state=random_state)


# In[64]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[65]:


train.to_csv(f'{data_path}CancerTrain.csv', index=False)


# In[66]:


train= train.copy()


# In[67]:


test.to_csv(f'{data_path}CancerTest.csv', index=False)


# In[68]:


test = test.copy()


# # Modelling

# In[69]:


X_train, y_train = dataset.drop('diagnosis',axis=1), dataset.diagnosis 


# In[70]:


total_time = 600
per_run_time_limit = 30


# In[71]:


le = preprocessing.LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)


# In[72]:


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=600,
    per_run_time_limit=30,
)


# In[73]:


automl.fit(X_train, y_train)


# In[76]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[77]:


logging.info(f'Saved classifier model at {model_path}model{timesstr}.pkl ')


# In[78]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


#profiler_data= PipelineProfiler.import_autosklearn(automl)
#PipelineProfiler.plot_pipeline_matrix(profiler_data)


# # Model Evluation and Explainability

# In[81]:


X_test, y_test = test.drop('diagnosis',axis=1), test.diagnosis 


# In[84]:


le = preprocessing.LabelEncoder()
le.fit(y_test)
y_test = le.transform(y_test)


# # Model Evaluation

# In[85]:


y_pred = automl.predict(X_test)


# In[86]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# In[89]:


dataset = pd.DataFrame(np.concatenate((X_test, y_test.reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


# dataset.columns = ['longitude', 'latitude', 'housing_median_age', 'households',
#                'median_income', 'bedroom_per_room',
#                'rooms_per_household', 'population_per_household', 'True Target', 'Predicted Target']


# In[ ]:


# fig = px.scatter(df, x='Predicted Target', y='True Target')
# fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


# logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# # Model Explanablity

# In[90]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[91]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
#shap.initjs()
#shap.force_plot(base_value = explainer.expected_value,
#                shap_values = shap_value_single,
 #               features = X_test.iloc[X_idx:X_idx+1,:], 
   #             show=False, matplotlib=True
             
    #            )
#plt.savefig(f"{model_path}shap_example_{timesstr}.png")
#logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[92]:


#shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
#shap.initjs()
#fig = shap.summary_plot(shap_values = shap_values,       features = X_test.iloc[0:50,:],         show=False)
#plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
#logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")


# --------------
# # End of This Notebook
