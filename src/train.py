#!/usr/bin/env python
# coding: utf-8

# <a id='Q0'></a>
# <center><a target="_blank" href="http://www.propulsion.academy"><img src="https://drive.google.com/uc?id=1McNxpNrSwfqu1w-QtlOmPSmfULvkkMQV" width="200" style="background:none; border:none; box-shadow:none;" /></a> </center>
# <center> <h4 style="color:#303030"> Applied Machine Learning: </h4> </center>
# <center> <h1 style="color:#303030">Predict House Value using neighboorhood characteristics</h1> </center>
# <p style="margin-bottom:1cm;"></p>
# <center style="color:#303030"><h4>Propulsion Academy, 2021</h4></center>
# <p style="margin-bottom:1cm;"></p>
# 
# <div style="background:#EEEDF5;border-top:0.1cm solid #EF475B;border-bottom:0.1cm solid #EF475B;">
#     <div style="margin-left: 0.5cm;margin-top: 0.5cm;margin-bottom: 0.5cm">
#         <p><strong>Goal:</strong>Implement Experiment tracking on a Regression problem using real estate data</p>
#         
# </div>
# 
# <nav style="text-align:right"><strong>
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/index.html" title="momentum"> Introduction to Data Science</a>|
#         <a style="color:#00BAE5" href="https://monolith.propulsion-home.ch/backend/api/momentum/materials/intro-2-ds-materials/weeks/week2/day2/index.html" title="momentum">Machine Learning Engineering</a>|
#         <a style="color:#00BAE5" href="https://colab.research.google.com/drive/15Ehfd97f7yTft_GT3GXsz3ZDSDS_dGYf?usp=sharing" title="momentum">  California House Price Training</a>
# </strong></nav>

# This notebook showcases a minimal implementation of training a machine learning model, with logging, 

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# ### package install

# Please note you **need to restart the run after the installation** for it to take effect!

# In[ ]:


# get_ipython().system(u'sudo apt-get install build-essential swig')
# get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
# get_ipython().system(u'pip install auto-sklearn')
# get_ipython().system(u'pip install pipelineprofiler # visualize the pipelines created by auto-sklearn')
# get_ipython().system(u'pip install shap')
# get_ipython().system(u'pip install --upgrade plotly')
# get_ipython().system(u'pip3 install -U scikit-learn')


# ### Packages imports

# In[ ]:


import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn import set_config
from sklearn.pipeline import Pipeline
from pandas_profiling import ProfileReport
from sklearn.impute import SimpleImputer

from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator, TransformerMixin
import autosklearn.regression
import PipelineProfiler

import plotly.express as px
import plotly.graph_objects as go

from joblib import dump

import shap

import datetime

import logging

import matplotlib.pyplot as plt


# ### Google Drive connection

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# ### options and settings

# In[ ]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/w2d2/data/raw/"


# In[ ]:


model_path = "/content/drive/MyDrive/Introduction2DataScience/tutorials/w2d2/models/"


# In[ ]:


timesstr = str(datetime.datetime.now()).replace(' ', '_')


# In[ ]:


logging.basicConfig(filename=f"{model_path}explog_{timesstr}.log", level=logging.INFO)


# Please Download the data from [this source](https://drive.google.com/file/d/1MUZrfW214Pv9p5cNjNNEEosiruIlLUXz/view?usp=sharing), and upload it on your Introduction2DataScience/data google drive folder.

# <a id='P1' name="P1"></a>
# ## [Loading Data and Train-Test Split](#P0)
# 

# In[ ]:


df = pd.read_csv(f'{data_path}california_housing.csv')


# In[ ]:


test_size = 0.2
random_state = 0


# In[ ]:


train, test = train_test_split(df, test_size=test_size, random_state=random_state)


# In[ ]:


logging.info(f'train test split with test_size={test_size} and random state={random_state}')


# In[ ]:


train.to_csv(f'{data_path}CaliforniaTrain.csv', index=False)


# In[ ]:


train= train.copy()


# In[ ]:


test.to_csv(f'{data_path}CaliforniaTest.csv', index=False)


# In[ ]:


test = test.copy()


# <a id='P2' name="P2"></a>
# ## [Modelling](#P0)

# In[ ]:


X_train, y_train = train.iloc[:,:-1], train.iloc[:,-1] 


# In[ ]:


total_time = 600
per_run_time_limit = 30


# In[ ]:


automl = autosklearn.regression.AutoSklearnRegressor(
    time_left_for_this_task=total_time,
    per_run_time_limit=per_run_time_limit,
)
automl.fit(X_train, y_train)


# In[ ]:


logging.info(f'Ran autosklearn regressor for a total time of {total_time} seconds, with a maximum of {per_run_time_limit} seconds per model run')


# In[ ]:


dump(automl, f'{model_path}model{timesstr}.pkl')


# In[ ]:


logging.info(f'Saved regressor model at {model_path}model{timesstr}.pkl ')


# In[ ]:


logging.info(f'autosklearn model statistics:')
logging.info(automl.sprint_statistics())


# In[ ]:


#profiler_data= PipelineProfiler.import_autosklearn(automl)
#PipelineProfiler.plot_pipeline_matrix(profiler_data)


# <a id='P2' name="P2"></a>
# ## [Model Evluation and Explainability](#P0)

# Let's separate our test dataframe into a feature variable (X_test), and a target variable (y_test):

# In[ ]:


X_test, y_test = test.iloc[:,:-1], test.iloc[:,-1]


# #### Model Evaluation

# Now, we can attempt to predict the median house value from our test set. To do that, we just use the .predict method on the object "automl" that we created and trained in the last sections:

# In[ ]:


y_pred = automl.predict(X_test)


# Let's now evaluate it using the mean_squared_error function from scikit learn:

# In[ ]:


logging.info(f"Mean Squared Error is {mean_squared_error(y_test, y_pred)}, \n R2 score is {automl.score(X_test, y_test)}")


# we can also plot the y_test vs y_pred scatter:

# In[ ]:


df = pd.DataFrame(np.concatenate((X_test, y_test.to_numpy().reshape(-1,1), y_pred.reshape(-1,1)),  axis=1))


# In[ ]:


df.columns = ['longitude', 'latitude', 'housing_median_age', 'households',
               'median_income', 'bedroom_per_room',
               'rooms_per_household', 'population_per_household', 'True Target', 'Predicted Target']


# In[ ]:


fig = px.scatter(df, x='Predicted Target', y='True Target')
fig.write_html(f"{model_path}residualfig_{timesstr}.html")


# In[ ]:


logging.info(f"Figure of residuals saved as {model_path}residualfig_{timesstr}.html")


# #### Model Explainability

# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test.iloc[X_idx:X_idx+1,:], 
                show=False,
                matplotlib=True
                )
plt.savefig(f"{model_path}shap_example_{timesstr}.png")
logging.info(f"Shapley example saved as {model_path}shap_example_{timesstr}.png")


# In[ ]:


shap_values = explainer.shap_values(X = X_test.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test.iloc[0:50,:],
                  show=False)
plt.savefig(f"{model_path}shap_summary_{timesstr}.png")
logging.info(f"Shapley summary saved as {model_path}shap_summary_{timesstr}.png")

