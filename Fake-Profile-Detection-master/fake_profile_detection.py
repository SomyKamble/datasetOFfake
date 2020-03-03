#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np 
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


dataset=pd.read_csv('Total_data.csv')


# In[3]:


dataset


# In[4]:


dataset=dataset.drop(columns=['Unnamed: 0','Unnamed: 0.1','Unnamed: 0.1.1'])


# In[5]:


dataset


# In[6]:


dataset=dataset.drop(columns=['UserID'])


# In[7]:


dataset1=dataset.drop(columns=['UserScreenName'])


# In[8]:


dataset1


# In[9]:


sns.pairplot(dataset1)


# In[9]:


dataset1.isna().sum()


# In[9]:


sns.distplot(dataset1['UserFollowersCount'])


# In[10]:


dataset1['UserFollowersCount'].max()


# In[11]:


from sklearn import preprocessing
import numpy as np


# In[10]:


y=dataset1['FakeOrNot']


# In[11]:


x=dataset1.drop(columns=['FakeOrNot'])


# In[12]:


from sklearn.preprocessing import LabelEncoder


# In[13]:


from sklearn.model_selection import train_test_split


#label encoding 


x['UserCreatedAt']=LabelEncoder().fit_transform(x['UserCreatedAt'])
x['UserLocation']=LabelEncoder().fit_transform(x['UserLocation'])
x['Current_Time']=LabelEncoder().fit_transform(x['Current_Time'])


# In[14]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)


# In[15]:


min_max_scaler = preprocessing.MinMaxScaler()


# In[18]:


x_train_minmax = min_max_scaler.fit_transform(x_train)


# In[15]:


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
model_log=LogisticRegression().fit(x_train,y_train)

model_knn=KNeighborsClassifier().fit(x_train,y_train)
model_tree=DecisionTreeClassifier().fit(x_train,y_train)


# In[16]:


pred=model_log.predict(x_test)
pred1=model_tree.predict(x_test)
pred2=model_knn.predict(x_test)


# In[17]:


from sklearn.metrics import confusion_matrix ,accuracy_score
confusion_matrix(y_test,pred)
accuracy_score(y_test,pred)


# In[18]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test,pred1)
accuracy_score(y_test,pred1)


# In[19]:



confusion_matrix(y_test,pred2)
accuracy_score(y_test,pred2)


# In[24]:


dataset1.describe()


# In[84]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
plt.figure(figsize=(30,20))
#tree.plot_tree(model_tree)
plot_tree(model_tree, filled=True)
plt.show()


# In[58]:


t=pd.DataFrame(columns=['colname'])
z={}

for i in range(len(dataset.columns)):
    typ=type(dataset.iloc[1,i])
    if(typ==str):
        
        print(dataset.iloc[:,i].name)
        z=dataset.iloc[:,i].name
        t=t.append({'colname':z},ignore_index=True)
        


# In[72]:


for i in range(len(t)):
    print(t.colname[i])
    v=pd.DataFrame({t.colname[i]:dataset[t.colname[i]]})


# In[100]:


#THChennai
c={}
typ=
g=dataset[dataset['UserLocation']=='India'].index
if len(g)>1:
    g=g[0]


# In[101]:


x['UserLocation'][g]#g=12


# In[124]:


x.iloc[:,4].name


# In[ ]:


c={}
typ=type(dataset[x.iloc[:,4].name][0])
typ=str(typ).split()[1]
print(typ)
c[x.iloc[:,4].name]=typ(input("please enter the "+x.iloc[:,4].name+"expected type is : "+typ))


# In[41]:


x=x.drop(columns=[x.iloc[:,1].name])


# In[50]:


typ=type(dataset[x.iloc[:,5].name][0])
typ
typ
x.iloc[:,4].name


# In[35]:


f={}
f[x.iloc[:,5].name]=9
x=x.drop(columns=[x.iloc[:,0].name])


# # code that works
# 

# In[24]:


h={}
def to_int(x):
    return int(x)
for i in range(len(x.columns)):
    typ=type(dataset[x.iloc[:,i].name][0])
    if(typ==str):
        h[x.iloc[:,i].name]=typ(input("please enter the "+x.iloc[:,i].name+"expected type is : "+str(typ)))    
        v=h[x.iloc[:,i].name]
        c=x.iloc[:,i].name
        k=h[x.iloc[:,i].name]
        print(k)
        g=dataset[dataset[c]==v].index
        if len(g)>1:
            g=g[0]
            k=x[x.iloc[:,i].name][g]
            
        else:
            k=x[x.iloc[:,i].name][g]
        h[c]=k
        to_int(h[c])
    else:
        h[x.iloc[:,i].name]=int(input("please enter the "+x.iloc[:,i].name+"expected type is : "+str(typ))) 
            
    
    


# In[27]:


s=pd.DataFrame(h)


# In[28]:


s


# In[29]:


fpred=model_tree.predict(s)


# In[35]:


fpred[0]
if(fpred[0]==1):
    print("The profile is fake")
else:
    print("The profile is not fake")


# In[ ]:




