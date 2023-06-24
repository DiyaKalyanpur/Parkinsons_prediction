#!/usr/bin/env python
# coding: utf-8

# In[46]:


#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[47]:


data = pd.read_csv("/Users/diyakalyanpur/Downloads/parkinsons.data")
data.head()


# In[48]:


data.shape


# In[49]:


data.isnull().sum() #no missing values


# In[50]:


#distribution
data["status"].value_counts() # 1 - parkinson's , 0 - healthy


# In[51]:


#train,test data
X = data.drop(columns=['name','status'],axis =1)
Y = data['status']
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size =0.2, random_state =3)
print(X_train)


# In[52]:


#Data Standardization
scaler = StandardScaler()
scaler.fit(X_train)


# In[53]:


X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)
print(X_train)


# In[54]:


#svm
model = svm.SVC(kernel = 'linear')
model.fit(X_train,Y_train)


# In[55]:


#Model Evaluation,accuracy score
X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(Y_train,X_train_prediction)
print(training_data_accuracy)


# In[56]:


X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(Y_test,X_test_prediction)
print(testing_data_accuracy)


# In[57]:


#predictive system 
input_data = [0.02924,0.04005,0.03772,0.08771,0.01353,20.64400,0.434969,0.819235,-4.117501,0.334147,2.405554,0.368975]
ans = np.asarray(input_data)
ans_reshape = ans.reshape(1,-1)
std_data = scaler.transform(ans_reshape)
model.predict(std_data)


# In[ ]:





# In[ ]:




