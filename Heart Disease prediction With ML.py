#!/usr/bin/env python
# coding: utf-8

# <b> Heart Disease prediction 

# In[84]:


#importing dependencies 

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[85]:


#Data collection and Precosseing

#convert data to pandas DataFrame
Heart_Data =pd.read_csv("/home/farhood/Desktop/heart_disease_data .csv")


# In[86]:


# print the data to obseeve

Heart_Data.head()


# In[87]:


Heart_Data.shape


# In[88]:


# information about the data

Heart_Data.info() #shows type of data and if there is any None(NULL)


# In[89]:


# check if there is any missing value

Heart_Data.isnull().sum() # show number of None(NULL) values


# In[90]:


# statical measurements

Heart_Data.describe()


# In[91]:


#checking the distribution of the Target Variable

Heart_Data['target'].value_counts()


# In[92]:


Heart_Data['fbs'].value_counts()


# <b> 1 == Defective Heart
#     
# <b> 0 == Healthy Heart

# <B> Spliting the Feature and target
# 
#  

# In[93]:


x = Heart_Data.drop(columns = 'target', axis =1) #whole Data without target coulmns
y = Heart_Data['target'] #only contains target columns


# In[94]:


# Spliting data into train and data

x_train, x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, stratify = y, random_state= 2)

# now one-third of the data is for test and two-third of the data is for training
# i tried 0.3 and 0.1 but it seems bst option is 0.2


# In[95]:


#Model_training with logsticregression

model = LogisticRegression()


# In[96]:


# training the model with training data

model.fit(x_train, y_train)


# In[97]:


# Model Evaluation 
# accuracy of the training model

x_train_prediction = model.predict(x_train)

training_data_accuracy = accuracy_score(x_train_prediction, y_train)

print(' accuracy of the trianing model: ', training_data_accuracy)


# In[98]:


# accuracy of the test model

x_test_prediction = model.predict(x_test)

test_data_accuracy = accuracy_score(x_test_prediction, y_test)

print(' accuracy of the test model: ', test_data_accuracy)


# <b> Note# the diffrence of accuraccy of the test and trianing model is very low which means this is great
# 
# 

# In[102]:


#create predictive system
input_data = (62,0,0,140,268,0,0,160,0,3.6,0,2,2)

# change the input data to a numpy array
input_data_as_numpy_array= np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]== 0):
  print('your heart is healthy')
else:
  print('bad news this data shows this preson has Defective heart')


# In[ ]:




