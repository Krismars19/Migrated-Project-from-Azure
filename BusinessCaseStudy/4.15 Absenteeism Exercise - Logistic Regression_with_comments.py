#!/usr/bin/env python
# coding: utf-8

# # Creating a logistic regression to predict absenteeism

# ## Import the relevant libraries

# In[1]:


# import the relevant libraries
import pandas as pd
import numpy as np


# ## Load the data

# In[2]:


# load the preprocessed CSV data
data_preprocessed = pd.read_csv('Absenteeism_preprocessed.csv')


# In[3]:


# eyeball the data
data_preprocessed.head()


# ## Create the targets

# In[4]:


# find the median of 'Absenteeism Time in Hours'
data_preprocessed['Absenteeism Time in Hours'].median()


# In[5]:


# create targets for our logistic regression
# they have to be categories and we must find a way to say if someone is 'being absent too much' or not
# what we've decided to do is to take the median of the dataset as a cut-off line
# in this way the dataset will be balanced (there will be roughly equal number of 0s and 1s for the logistic regression)
# as balancing is a great problem for ML, this will work great for us
# alternatively, if we had more data, we could have found other ways to deal with the issue 
# for instance, we could have assigned some arbitrary value as a cut-off line, instead of the median

# note that what line does is to assign 1 to anyone who has been absent 4 hours or more (more than 3 hours)
# that is the equivalent of taking half a day off

# initial code from the lecture
# targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 3, 1, 0)

# parameterized code
targets = np.where(data_preprocessed['Absenteeism Time in Hours'] > 
                   data_preprocessed['Absenteeism Time in Hours'].median(), 1, 0)


# In[6]:


# eyeball the targets
targets


# In[7]:


# create a Series in the original data frame that will contain the targets for the regression
data_preprocessed['Excessive Absenteeism'] = targets


# In[8]:


# check what happened
# maybe manually see how the targets were created
data_preprocessed.head()


# ## A comment on the targets

# In[9]:


# check if dataset is balanced (what % of targets are 1s)
# targets.sum() will give us the number of 1s that there are
# the shape[0] will give us the length of the targets array
targets.sum() / targets.shape[0]


# In[10]:


# create a checkpoint by dropping the unnecessary variables
# also drop the variables we 'eliminated' after exploring the weights
data_with_targets = data_preprocessed.drop(['Absenteeism Time in Hours','Day of the Week',
                                            'Daily Work Load Average','Distance to Work'],axis=1)


# In[11]:


# check if the line above is a checkpoint :)

# if data_with_targets is data_preprocessed = True, then the two are pointing to the same object
# if it is False, then the two variables are completely different and this is in fact a checkpoint
data_with_targets is data_preprocessed


# In[12]:


# check what's inside
data_with_targets.head()


# ## Select the inputs for the regression

# In[13]:


data_with_targets.shape


# In[14]:


# Selects all rows and all columns until 14 (excluding)
data_with_targets.iloc[:,:14]


# In[15]:


# Selects all rows and all columns but the last one (basically the same operation)
data_with_targets.iloc[:,:-1]


# In[16]:


# Create a variable that will contain the inputs (everything without the targets)
unscaled_inputs = data_with_targets.iloc[:,:-1]


# ## Standardize the data

# In[17]:


# standardize the inputs

# standardization is one of the most common preprocessing tools
# since data of different magnitude (scale) can be biased towards high values,
# we want all inputs to be of similar magnitude
# this is a peculiarity of machine learning in general - most (but not all) algorithms do badly with unscaled data

# a very useful module we can use is StandardScaler 
# it has much more capabilities than the straightforward 'preprocessing' method
from sklearn.preprocessing import StandardScaler


# we will create a variable that will contain the scaling information for this particular dataset
# here's the full documentation: http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html

# define scaler as an object
absenteeism_scaler = StandardScaler()


# In[18]:


# import the libraries needed to create the Custom Scaler
# note that all of them are a part of the sklearn package
# moreover, one of them is actually the StandardScaler module, 
# so you can imagine that the Custom Scaler is build on it

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler

# create the Custom Scaler class

class CustomScaler(BaseEstimator,TransformerMixin): 
    
    # init or what information we need to declare a CustomScaler object
    # and what is calculated/declared as we do
    
    def __init__(self,columns,copy=True,with_mean=True,with_std=True):
        
        # scaler is nothing but a Standard Scaler object
        self.scaler = StandardScaler(copy,with_mean,with_std)
        # with some columns 'twist'
        self.columns = columns
        self.mean_ = None
        self.var_ = None
        
    
    # the fit method, which, again based on StandardScale
    
    def fit(self, X, y=None):
        self.scaler.fit(X[self.columns], y)
        self.mean_ = np.mean(X[self.columns])
        self.var_ = np.var(X[self.columns])
        return self
    
    # the transform method which does the actual scaling

    def transform(self, X, y=None, copy=None):
        
        # record the initial order of the columns
        init_col_order = X.columns
        
        # scale all features that you chose when creating the instance of the class
        X_scaled = pd.DataFrame(self.scaler.transform(X[self.columns]), columns=self.columns)
        
        # declare a variable containing all information that was not scaled
        X_not_scaled = X.loc[:,~X.columns.isin(self.columns)]
        
        # return a data frame which contains all scaled features and all 'not scaled' features
        # use the original order (that you recorded in the beginning)
        return pd.concat([X_not_scaled, X_scaled], axis=1)[init_col_order]


# In[19]:


# check what are all columns that we've got
unscaled_inputs.columns.values


# In[20]:


# choose the columns to scale
# we later augmented this code and put it in comments
# columns_to_scale = ['Month Value','Day of the Week', 'Transportation Expense', 'Distance to Work',
       #'Age', 'Daily Work Load Average', 'Body Mass Index', 'Children', 'Pet']
    
# select the columns to omit
columns_to_omit = ['Reason_1', 'Reason_2', 'Reason_3', 'Reason_4','Education']


# In[21]:


# create the columns to scale, based on the columns to omit
# use list comprehension to iterate over the list
columns_to_scale = [x for x in unscaled_inputs.columns.values if x not in columns_to_omit]


# In[22]:


# declare a scaler object, specifying the columns you want to scale
absenteeism_scaler = CustomScaler(columns_to_scale)


# In[23]:


# fit the data (calculate mean and standard deviation); they are automatically stored inside the object 
absenteeism_scaler.fit(unscaled_inputs)


# In[24]:


# standardizes the data, using the transform method 
# in the last line, we fitted the data - in other words
# we found the internal parameters of a model that will be used to transform data. 
# transforming applies these parameters to our data
# note that when you get new data, you can just call 'scaler' again and transform it in the same way as now
scaled_inputs = absenteeism_scaler.transform(unscaled_inputs)


# In[25]:


# the scaled_inputs are now an ndarray, because sklearn works with ndarrays
scaled_inputs


# In[26]:


# check the shape of the inputs
scaled_inputs.shape


# ## Split the data into train &amp; test and shuffle

# ### Import the relevant module

# In[27]:


# import train_test_split so we can split our data into train and test
from sklearn.model_selection import train_test_split


# ### Split

# In[28]:


# check how this method works
train_test_split(scaled_inputs, targets)


# In[29]:


# declare 4 variables for the split
x_train, x_test, y_train, y_test = train_test_split(scaled_inputs, targets, #train_size = 0.8, 
                                                                            test_size = 0.2, random_state = 20)


# In[30]:


# check the shape of the train inputs and targets
print (x_train.shape, y_train.shape)


# In[31]:


# check the shape of the test inputs and targets
print (x_test.shape, y_test.shape)


# ## Logistic regression with sklearn

# In[32]:


# import the LogReg model from sklearn
from sklearn.linear_model import LogisticRegression

# import the 'metrics' module, which includes important metrics we may want to use
from sklearn import metrics


# ### Training the model

# In[33]:


# create a logistic regression object
reg = LogisticRegression()


# In[34]:


# fit our train inputs
# that is basically the whole training part of the machine learning
reg.fit(x_train,y_train)


# In[35]:


# assess the train accuracy of the model
reg.score(x_train,y_train)


# ### Manually check the accuracy

# In[36]:


# find the model outputs according to our model
model_outputs = reg.predict(x_train)
model_outputs


# In[37]:


# compare them with the targets
y_train


# In[38]:


# ACTUALLY compare the two variables
model_outputs == y_train


# In[39]:


# find out in how many instances we predicted correctly
np.sum((model_outputs==y_train))


# In[40]:


# get the total number of instances
model_outputs.shape[0]


# In[41]:


# calculate the accuracy of the model
np.sum((model_outputs==y_train)) / model_outputs.shape[0]


# ### Finding the intercept and coefficients

# In[42]:


# get the intercept (bias) of our model
reg.intercept_


# In[43]:


# get the coefficients (weights) of our model
reg.coef_


# In[44]:


# check what were the names of our columns
unscaled_inputs.columns.values


# In[45]:


# save the names of the columns in an ad-hoc variable
feature_name = unscaled_inputs.columns.values


# In[46]:


# use the coefficients from this table (they will be exported later and will be used in Tableau)
# transpose the model coefficients (model.coef_) and throws them into a df (a vertical organization, so that they can be
# multiplied by certain matrices later) 
summary_table = pd.DataFrame (columns=['Feature name'], data = feature_name)

# add the coefficient values to the summary table
summary_table['Coefficient'] = np.transpose(reg.coef_)

# display the summary table
summary_table


# In[47]:


# do a little Python trick to move the intercept to the top of the summary table
# move all indices by 1
summary_table.index = summary_table.index + 1

# add the intercept at index 0
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]

# sort the df by index
summary_table = summary_table.sort_index()
summary_table


# ## Interpreting the coefficients

# In[48]:


# create a new Series called: 'Odds ratio' which will show the.. odds ratio of each feature
summary_table['Odds_ratio'] = np.exp(summary_table.Coefficient)


# In[49]:


# display the df
summary_table


# In[50]:


# sort the table according to odds ratio
# note that by default, the sort_values method sorts values by 'ascending'
summary_table.sort_values('Odds_ratio', ascending=False)


# ## Testing the model

# In[51]:


# assess the test accuracy of the model
reg.score(x_test,y_test)


# In[52]:


# find the predicted probabilities of each class
# the first column shows the probability of a particular observation to be 0, while the second one - to be 1
predicted_proba = reg.predict_proba(x_test)

# let's check that out
predicted_proba


# In[53]:


predicted_proba.shape


# In[54]:


# select ONLY the probabilities referring to 1s
predicted_proba[:,1]


# ## Save the model

# In[55]:


# import the relevant module
import pickle


# In[56]:


# pickle the model file
with open('model', 'wb') as file:
    pickle.dump(reg, file)


# In[57]:


# pickle the scaler file
with open('scaler','wb') as file:
    pickle.dump(absenteeism_scaler, file)

