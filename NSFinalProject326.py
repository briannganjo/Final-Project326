#!/usr/bin/env python
# coding: utf-8

# In[ ]:


########################
#       Brian          #
#      Carter          #
#       Brett          #
#       Malik          #
#     INST 326         #
#     Final Project    # 
#      12/12           #
########################


# In[1]:


#import the pandas commands
#importing the pandas library
import os 
import csv
import pandas as pd 
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


df = pd.read_csv("Automobile_data.csv")  #this is the data frame handler which will carry the csv file


# In[3]:


df.head() #this is the header for the car dataset displaying variables first 5


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline  #Importin the matplot displayment package used for the plots')


# In[5]:


df.dtypes  #this is telling us the types for the dataset


# In[6]:


df['num-of-doors'].value_counts()  #this will display the value counts for the variabels number of doors


# In[7]:


pd.to_numeric(df['price'])  #this is converting the values within the price variables to numeric values from objects


# In[8]:


df.isnull().sum() #this is checking for missing variables withtin the dataset


# In[9]:


# this is the code for a box plot displayment of the analysis for the project
# the horsepower variable is going to be the x/independent variable in our boxplot
#the price of the automobiles will be the y/dependent variable in our box plot
sns.boxplot(x='city-mpg',y='price', data=df)


# In[10]:


sns.regplot(x='city-mpg', y='price', data=df)  #this is a linear regression plot for the linear regression graph it will display correlation between city-mpg and the price


# In[11]:


df.describe()  #this is a statistical description of the dataset we will be using for the analysis


# In[12]:


sns.jointplot(x='city-mpg', y='price', data=df) #this is a marginal distribution plot as well as a scatter plot displaying the correlation between the variables


# In[13]:


sns.distplot(df['city-mpg'])  #this will display a curve spread of the data over a marginal distribution


# In[16]:


import statsmodels.formula.api as smf #this is importing the package for the linear analysis


# In[17]:


lm = smf.ols(formula = 'price ~ horsepower', data=df).fit()  #this is the first linear analysis which will conducted for the price and horse power of the dataset

print(lm.summary())


# In[14]:


#using scikit-learn:
from sklearn import linear_model #this will display the coeffecient and the intercepts for the analysis

est = linear_model.LinearRegression(fit_intercept = True)
#create estimator object
est.fit(df[['city-mpg']],
df[['price']])
#print result
print("Coef:", est.coef_, "\nIntercept:", est.intercept_)


# In[15]:


from scipy import stats  #this is importing the pacgake for the t test
df_2 = df[ df['num-of-doors'] == 'two']['city-mpg']  #we will be comparing the cit mpg in accord to four door vs a two door
df_4 = df[ df['num-of-doors'] == 'four']['city-mpg']
stats.ttest_ind(df_2, df_4)


# In[ ]:




