#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# ### Repeating steps from Round 3 

# In[2]:


df = pd.read_csv(r'C:\Users\galrodri\Documents\GitHub\lab-customer-analysis-round-4\files_for_lab\csv_files\marketing_customer_analysis.csv')
df.head()


# In[3]:


df.columns = map(lambda x: x.lower().replace("-", "_").replace(" ", "_"), df.columns) # lowering headers


# In[4]:


df.isna().sum()


# In[5]:


df['effective_to_date'] = df['effective_to_date'].astype('datetime64[ns]')
df['effective_to_date']


# In[6]:


df.dtypes


# In[7]:


numericals = df.select_dtypes(np.number)
numericals.head()


# In[8]:


categoricals = df.select_dtypes(np.object)
categoricals.head()


# In[9]:


for col in numericals.columns:
    sns.displot(data=numericals, x=col)
    plt.show()


# In[10]:


for col in numericals.columns:
    plt.hist(df[col])
    plt.show()


# In[11]:


numericals.corr()


# In[12]:


numericals.corr()['total_claim_amount']


# In[13]:


corr_matrix = numericals.drop('total_claim_amount',axis=1).corr()
corr_matrix


# In[14]:


high_correlation = corr_matrix > 0.9
high_correlation


# In[15]:


sns_plot = sns.heatmap(corr_matrix, annot=True)
plt.figure(figsize=(16, 6))
plt.show()


# ## Round 5

# In[16]:


# Question 1: X-Y split

x = df.drop(['total_claim_amount','effective_to_date'], axis=1) 
y = df.total_claim_amount # variable that we want to predict


# In[17]:


# Question 2: Splitting categorical and numerical variables
x_num = x.select_dtypes(np.number)
x_cat = x.drop(columns = x_num.columns)
x_num.head()


# In[18]:


x_cat.head()


# In[19]:


x_num.describe().transpose()


# In[20]:


# normalize (standard)
from sklearn.preprocessing import StandardScaler

X_scaled=StandardScaler().fit_transform(x_num)

def normalize(X):
    X_mean=X.mean(axis=0)
    X_std=X.std(axis=0)
    X_std[X_std==0]=1.0
    X=(X-X_mean)/X_std
    return X

X_num=normalize(x_num)
X_num.head()


# In[21]:


# removing customer from categorical variables
cust = x_cat['customer']
x_cat_nocust = x_cat.drop('customer', axis = 1)
# Converting categorical to dummy variables
dummy = pd.get_dummies(x_cat_nocust)
data_cat = dummy
data_cat.head(1)


# In[22]:


# combining numerical data with dummify variables
final_df = pd.concat([X_num,data_cat], axis = 1)


# In[23]:


# linear regression model
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

X = final_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
model
predictions  = lm.predict(X_test)


# In[24]:


print("R2 for the model is: ",round(r2_score(y_test, predictions),3))
print("MSE for the model is: ",round(mean_squared_error(y_test, predictions),3))
print("The RMSE of the linear model is: ",round(mean_squared_error(y_test, predictions,squared = False),3))
print("The MAE of the linear model is: ",round(mean_absolute_error(y_test, predictions),3))


# In[28]:


# improve the model
final_df = final_df.drop(['sales_channel_Branch', 'sales_channel_Call Center','sales_channel_Web'], axis=1) 
X = final_df
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=100)
lm = linear_model.LinearRegression()
model = lm.fit(X_train,y_train)
model
predictions  = lm.predict(X_test)


# In[29]:


print("R2 for the model is: ",round(r2_score(y_test, predictions),3))
print("MSE for the model is: ",round(mean_squared_error(y_test, predictions),3))
print("The RMSE of the linear model is: ",round(mean_squared_error(y_test, predictions,squared = False),3))
print("The MAE of the linear model is: ",round(mean_absolute_error(y_test, predictions),3))

