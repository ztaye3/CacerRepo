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

# <a id='SU' name="SU"></a>
# ## [Set up](#P0)

# **Package install**

# In[ ]:


#get_ipython().system(u'sudo apt-get install cookiecutter')
#get_ipython().system(u'pip install gdown')
#get_ipython().system(u'pip install dvc')
#get_ipython().system(u"pip install 'dvc[gdrive]'")


# In[ ]:


#get_ipython().system(u'sudo apt-get install build-essential swig')
#get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system(u'pip install auto-sklearn==0.12.5')


# In[ ]:


#get_ipython().system(u'pip install joblib')


# In[ ]:


#get_ipython().system(u'pip install pipelineprofiler')


# In[ ]:


#get_ipython().system(u'pip install shap')


# In[ ]:


#get_ipython().system(u'pip install --upgrade plotly')


# In[ ]:


#get_ipython().system(u'pip3 install -U scikit-learn')


# In[ ]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from pandas_profiling import ProfileReport
import matplotlib.pyplot as plt
import plotly
plotly.__version__

import plotly.graph_objects as go
import plotly.io as pio
import plotly.express as px
from plotly.subplots import make_subplots
import logging
import joblib


# In[ ]:


from sklearn.preprocessing import StandardScaler, OrdinalEncoder, OneHotEncoder
from sklearn import set_config
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn import preprocessing


# In[ ]:


import autosklearn.classification
import PipelineProfiler


# In[ ]:


logging.basicConfig(filename = 'logs.log' , level = logging.INFO)


# In[ ]:


import shap


# Connect to your Google Drive

# In[ ]:


from google.colab import drive
drive.mount('/content/drive', force_remount=True)


# In[ ]:


data_path = "/content/drive/MyDrive/Introduction2DataScience/data/"


# In[ ]:


pd.set_option('display.max_rows', 20)


# In[ ]:


set_config(display='diagram')


# In[ ]:


#get_ipython().magic(u'matplotlib inline')


# _Your Comments here_

# ### Data Structure and types

# **Load the csv file as a DataFrame using Pandas**

# In[ ]:


dataset = pd.read_csv(f'{data_path}data-breast-cancer.csv')


# In[ ]:


logging.info("Read the dataset")


# **Observation:**
# 
# 1.   Drop column 'Unnamed: 32' since it's all values are empty:
# 2.   Drop column 'Id' since it doesn't contain a specific information about the cancer cell.

# In[ ]:


drop_column = ['id','diagnosis']


# In[ ]:


X = dataset.drop(drop_column , axis = 1)
y = dataset.diagnosis
num_variables = ['radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean',
       'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']


# 
# _Your Comments here_

# We now can do the test:train split

# In[ ]:


X_train, X_test, y_train, y_test = train_test_split(X, # original dataframe to be split
                                                     y,
                                                     test_size=0.2, # proportion of the rows to put in the test set
                                                     stratify=y,
                                                     random_state=42) # for reproducibility (see explanation below)


# In[ ]:


logging.info("Successfully divided data into training and testing set")


# ### Pipeline Definition

# In[ ]:


numeric_transformer = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value=0)),
                                      ('scaler', StandardScaler())])


# In[ ]:


ohe_transformer = OneHotEncoder(handle_unknown='ignore')


# In[ ]:


preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_variables)
        ])


# In[ ]:


classification_model = Pipeline(steps=[('preprocessor', preprocessor),
                                          ('classifier', LogisticRegression())])


# In[ ]:


classification_model


# In[ ]:


logging.info("Model prepared")


# _Your Comments here_

# ### Model Training

# In[ ]:


classification_model.fit(X_train, y_train)


# In[ ]:


col_names = num_variables.copy()


# In[ ]:


X_train_encoded = pd.DataFrame(classification_model['preprocessor'].transform(X_train), columns=col_names)


# Encode feature 'diagnosis' with label encoder

# In[ ]:


le = preprocessing.LabelEncoder()
le.fit(y)
y_train_encoded = le.transform(y_train)


# In[ ]:


automl = autosklearn.classification.AutoSklearnClassifier(
    time_left_for_this_task=60,
    per_run_time_limit=30,
)


# In[ ]:


automl.fit(X_train_encoded, y_train_encoded)


# In[ ]:


profiler_data= PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# In[ ]:


profiler_data= PipelineProfiler.import_autosklearn(automl)
PipelineProfiler.plot_pipeline_matrix(profiler_data)


# In[ ]:


logging.info("Model finally trained")


# Now, we save the trained model using joblib.

# In[ ]:


joblib.dump(classification_model , "mlmodel")


# In[ ]:


logging.info("Model finally dumped")


# ### Model Evaluation

# In[ ]:


# your code here


# In[ ]:


X_test_encoded = pd.DataFrame(classification_model['preprocessor'].transform(X_test), columns=col_names)


# In[ ]:


y_pred = automl.predict(X_test_encoded)


# In[ ]:


y_test_encoded = le.transform(y_test)


# In[ ]:


confusion_matrix(y_test_encoded,y_pred)


# In[ ]:


ConfusionMatrixDisplay(confusion_matrix(y_test_encoded,y_pred))


# In[ ]:


explainer = shap.KernelExplainer(model = automl.predict, data = X_test_encoded.iloc[:50, :], link = "identity")


# In[ ]:


# Set the index of the specific example to explain
X_idx = 0
shap_value_single = explainer.shap_values(X = X_test_encoded.iloc[X_idx:X_idx+1,:], nsamples = 100)
X_test.iloc[X_idx:X_idx+1,:]
# print the JS visualization code to the notebook
shap.initjs()
shap.force_plot(base_value = explainer.expected_value,
                shap_values = shap_value_single,
                features = X_test_encoded.iloc[X_idx:X_idx+1,:]
                )


# In[ ]:


shap_values = explainer.shap_values(X = X_test_encoded.iloc[0:50,:], nsamples = 100)


# In[ ]:


# print the JS visualization code to the notebook
shap.initjs()
fig = shap.summary_plot(shap_values = shap_values,
                  features = X_test_encoded.iloc[0:50,:]
                  )
plt.savefig(f"shap_summary.png")


# In[ ]:


logging.info("Model evaluated")


# # Create a cookiecutter data science project directory in your google drive and track its evolution using git,

# Mount drive:

# In[ ]:


from google.colab import drive

drive.mount('/content/drive', force_remount=True)


#  Move to the directory of the data:

# In[ ]:


#get_ipython().magic(u'cd /content/drive/MyDrive/Introduction2DataScience/data')


# Open project with following parameters:
# 
# > Indented block
# project_name [project_name]: cancer
# repo_name [cancer]: CancerRepo
# author_name [Your name (or your organization/company/team)]: ztaye3@gmail.com
# 
# 

# In[ ]:


#get_ipython().system(u'cookiecutter https://github.com/drivendata/cookiecutter-data-sciences')


# Let's checkout its structure:

# In[ ]:


#get_ipython().magic(u'cd CancerRepo')
#get_ipython().system(u'ls')


# **Track code using git:**

# In[ ]:


#get_ipython().system(u'git init')


# In[ ]:


#get_ipython().system(u'git add .')


# In[ ]:


#get_ipython().system(u'git config --global user.email "ztaye3@gmail.com"')
#get_ipython().system(u'git config --global user.name "ztaye"')


# In[ ]:


#get_ipython().system(u'git status')


# In[ ]:


# !git commit -m "cookiecutter data science project structure"


# # Place your raw data and your machine learning notebooks in the dedicated folders

# In[ ]:


#get_ipython().magic(u'cd data/raw')


# In[ ]:


#get_ipython().system(u'gdown https://drive.google.com/uc?id=1af2YyHIp__OdpuUeOZFwmwOvCsS0Arla')


# In[ ]:


#get_ipython().magic(u'cd ../../notebooks/')


# In[ ]:


#get_ipython().system(u'gdown https://drive.google.com/uc?id=15Ehfd97f7yTft_GT3GXsz3ZDSDS_dGYf')


# # Convert your notebook into a python script and place it in the dedicated folder

# _Your Comments here_

# **Convert Notebook to script**

# In[ ]:


#get_ipython().magic(u'cd /content/drive/My Drive/Introduction2DataScience/data/CancerRepo/notebooks')


# In[ ]:


#get_ipython().system(u'jupyter nbconvert --to script --output ../src/train SIT_W2D2_HT2_Model_Development.ipynb --to python')


# In[ ]:


#get_ipython().magic(u'cd /content/drive/My Drive/Introduction2DataScience/data/CancerRepo/src/')


# # Modify the python script and execute it from the command line (use !python myscript.py in another notebook)

# Install script dependecies

# In[ ]:


#get_ipython().system(u'sudo apt-get install build-essential swig')
#get_ipython().system(u'curl https://raw.githubusercontent.com/automl/auto-sklearn/master/requirements.txt | xargs -n 1 -L 1 pip install')
#get_ipython().system(u'pip install auto-sklearn')
#get_ipython().system(u'pip install pipelineprofiler # visualize the pipelines created by auto-sklearn')
#get_ipython().system(u'pip install shap')
#get_ipython().system(u'pip install --upgrade plotly')
#get_ipython().system(u'pip3 install -U scikit-learn')


# Execute the script:

# In[ ]:


#get_ipython().system(u'python train.py ')


# --------------
# # End of This Notebook
