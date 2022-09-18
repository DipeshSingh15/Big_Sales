#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns


# In[2]:


df=pd.read_csv("C:\\Users\\Dipesh Singh\\Downloads\\Train-Set.csv")


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


sns.heatmap(df.corr())
plt.show()


# In[7]:


df.describe()


# In[8]:


df.info()


# In[9]:


df["FatContent"].value_counts()


# In[10]:


df=df.replace(to_replace="low fat",value="Low Fat")
#df=df.repalce(to_replace="LF",value="Low Fat")
df["FatContent"].value_counts()


# In[11]:


df=df.replace(to_replace="LF",value="Low Fat")
df["FatContent"].value_counts()


# In[12]:


df=df.replace(to_replace="reg",value="Regular")
df["FatContent"].value_counts()


# In[13]:


df=df.replace(to_replace="Low Fat",value=1)
df["FatContent"].value_counts()


# In[14]:


df=df.replace(to_replace="Regular",value=0)
df["FatContent"].value_counts()


# In[15]:


df.info()


# In[16]:


df["ProductType"].value_counts()


# In[17]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[18]:


le.fit(df["ProductType"])
le.transform(df["ProductType"])


# In[19]:


df["ProductType"].value_counts()


# In[20]:


df.corr()


# In[21]:


df["Weight"]


# In[ ]:





# In[22]:


mean=df["Weight"].mean()
mean


# In[23]:


df["Weight"].fillna(df["Weight"].mean(),inplace=True)


# In[24]:


df["OutletSize"].value_counts()


# In[25]:


df["OutletSize"].mode()


# In[26]:


df["OutletSize"].fillna(df["OutletSize"].mode(),inplace=True)
df.isnull().sum()


# In[27]:


le.fit(df["OutletSize"])
le.transform(df["OutletSize"])


# In[28]:


df["OutletSize"]


# In[29]:


from scipy.stats import mode
mode_size = df.pivot_table(values = 'OutletSize', index = 'OutletType',aggfunc=(lambda x : mode(x.dropna()).mode[0]))


# In[30]:


mode_size


# In[31]:


null=df["OutletSize"].isnull()


# In[32]:


df.loc[null, 'OutletSize'] = df.loc[null, 'OutletType'].apply(lambda x: mode_size.loc[x])


# In[33]:


df.isnull().sum()


# 
# # Data Analysis

# In[34]:


plt.figure(figsize=(8,8))
sns.distplot(df["Weight"],color="blue")
plt.show()


# In[35]:


sns.countplot(df["FatContent"])
plt.show()


# In[36]:


sns.distplot(df["ProductVisibility"])
plt.show()


# In[37]:


sns.distplot(df["MRP"])
plt.show()


# In[38]:


sns.countplot(df["EstablishmentYear"])
plt.show()


# In[39]:


sns.countplot(df["OutletSize"])
plt.show()


# In[40]:


plt.figure(figsize=(30,9))
sns.countplot(df["ProductType"])
plt.show()


# # Data Preprocessing

# In[41]:


df.info()


# In[42]:


df["ProductID"]=le.fit_transform(df["ProductID"])
df["ProductType"]=le.fit_transform(df["ProductType"])
df["OutletID"]=le.fit_transform(df["OutletID"])
df["OutletSize"]=le.fit_transform(df["OutletSize"])
df["LocationType"]=le.fit_transform(df["LocationType"])
df["OutletType"]=le.fit_transform(df["OutletType"])


# In[43]:


df.info()


# In[44]:


df.head()


# In[45]:


x=df.drop(columns="OutletSales",axis=1)
y=df["OutletSales"]


# In[46]:


x


# In[47]:


from sklearn.model_selection import train_test_split


# In[48]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)


# # Machine Learning Model Training

# In[49]:


from sklearn.linear_model import LinearRegression
lr=LinearRegression()


# In[50]:


lr.fit(x_train,y_train)


# In[51]:


y_pred=lr.predict(x_test)


# In[52]:


#y_pred


# In[53]:


from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
mse=mean_squared_error(y_test,y_pred)
ae=mean_absolute_error(y_test,y_pred)
re=r2_score(y_test,y_pred)
print(mse)
print(ae)
print(re)


# In[54]:


from sklearn.ensemble import RandomForestRegressor


# In[55]:


rg=RandomForestRegressor()


# In[56]:


rg.fit(x_train,y_train)


# In[57]:


y_pred2=rg.predict(x_test)


# In[58]:


mse1=mean_squared_error(y_test,y_pred2)
ae1=mean_absolute_error(y_test,y_pred2)
re1=r2_score(y_test,y_pred2)
print(mse1)
print(ae1)
print(re1)


# In[59]:


df_test=pd.read_csv("C:\\Users\\Dipesh Singh\\Downloads\\Test-Set.csv")
df_test.head()


# In[60]:


df_test.shape


# In[61]:


df_test.info()


# In[62]:


df_test["Weight"].fillna(df_test["Weight"].mean(),inplace=True)


# In[63]:


df_test.info()


# In[64]:


mode_size1 = df_test.pivot_table(values = 'OutletSize', index = 'OutletType',aggfunc=(lambda x : mode(x.dropna()).mode[0]))


# In[65]:


null1=df_test["OutletSize"].isnull()


# In[67]:


df_test.loc[null1, 'OutletSize'] = df_test.loc[null1, 'OutletType'].apply(lambda x: mode_size.loc[x])


# In[68]:


df_test.info()


# In[69]:


df_test["ProductID"]=le.fit_transform(df_test["ProductID"])
df_test["ProductType"]=le.fit_transform(df_test["ProductType"])
df_test["OutletID"]=le.fit_transform(df_test["OutletID"])
df_test["OutletSize"]=le.fit_transform(df_test["OutletSize"])
df_test["LocationType"]=le.fit_transform(df_test["LocationType"])
df_test["OutletType"]=le.fit_transform(df_test["OutletType"])


# In[72]:


df_test=df_test.replace(to_replace="low fat",value="Low Fat")
#df=df.repalce(to_replace="LF",value="Low Fat")
df_test["FatContent"].value_counts()


# In[73]:


df_test=df_test.replace(to_replace="LF",value="Low Fat")
df_test["FatContent"].value_counts()A


# In[74]:


df_test=df_test.replace(to_replace="reg",value="Regular")
df_test["FatContent"].value_counts()


# In[75]:


df_test["FatContent"]=le.fit_transform(df_test["FatContent"])


# In[76]:


df_test.info()


# In[77]:


y_pred3=rg.predict(df_test)


# In[78]:


y_pred3


# In[79]:


pred=y_pred3
res=pd.DataFrame(pred,columns=["OutletSales"])


# In[80]:


res


# In[ ]:




