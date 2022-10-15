#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from sklearn.metrics import classification_report  
from sklearn import tree


# In[ ]:


#A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
#Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) 
#& all other variable will be independent in the analysis.  


# In[2]:


df = pd.read_csv("C:\\Users\\HP\\Documents\\Excel R data\\Company_Data.csv")
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


sns.pairplot(df)


# # 1a. Dividing Sales in Three Categories (Low, Med, High)
#     # Spliting variables in Train and Test

# In[7]:


# converting sales value in catagorical data
Sales_C = []
for Sales in df["Sales"]:
    if Sales<=5:
        Sales_C.append("low")
    elif Sales<=12:
        Sales_C.append("Med")
    else:
      Sales_C.append("high")
df["Sales_C"]= Sales_C
df1 = df
df1.head()


# In[8]:


print(df1.Sales_C.unique())
print(df1.Sales_C.value_counts())
print(df1.ShelveLoc.value_counts())
print(df1.Urban.value_counts())
print(df1.US.value_counts())


# In[9]:


sns.countplot(df1.ShelveLoc)


# In[10]:


sns.countplot(df1.Urban)


# In[7]:


sns.countplot(df1.US)


# In[8]:


sns.countplot(df1.Sales_C)


# In[11]:


from sklearn.preprocessing import LabelEncoder
LE = LabelEncoder()
df1["ShelveLoc"]= LE.fit_transform(df1["ShelveLoc"])
df1["Urban"] = LE.fit_transform(df1["Urban"])
df1["US"] = LE.fit_transform(df1["US"])
print(df1.head())
print(df1.shape)


# In[12]:


# Correlation analysis for data
corr = df1.corr()
#Plot figsize
fig, ax = plt.subplots(figsize=(10, 6))
#Generate Heat Map, allow annotations and place floats in map
sns.heatmap(corr, cmap='coolwarm',linewidths = 1,linecolor="y", annot=True, fmt=".4f")
plt.show()


# # selection of X and Y variables

# In[13]:


X = df1.iloc[:,1:11]
print(list(X))
Y = df1.iloc[:,11]


# In[14]:


pd.crosstab(Y,Y)


# # Entropy method

# In[15]:


#Entropy 

# SPLITING DATA IN TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
ET.fit(X_train, Y_train) 

# Predicted Y, Confusion Matrix and Accuracy score for TRAINNING data
#Y1_pred = ET.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,Y1_pred)
#print(cm)
#ac = accuracy_score(Y_train,Y1_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
Y_pred = ET.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
ac = accuracy_score(Y_test,Y_pred)
print(ac)


# # Classification Report

# In[16]:


from sklearn.metrics import classification_report  
print(classification_report(Y_pred, Y_test)) 


#  # Using KFold

# In[17]:


# Using KFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 7, shuffle = True)
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
from sklearn.model_selection import cross_val_score
results = cross_val_score(ET, X, Y, cv=kfold)
print(results)
print(np.mean(abs(results)))
# Predicted Y, Confusion Matrix and Accuracy score for TEST data
from sklearn.model_selection import KFold, cross_val_predict
Y1_pred = cross_val_predict(ET, X, Y, cv=kfold)
print(pd.crosstab(Y,Y))
print(pd.crosstab(Y1_pred,Y1_pred))
print(pd.crosstab(Y,Y1_pred))


# In[18]:


from sklearn.metrics import classification_report  
print(classification_report(Y1_pred, Y)) 


# In[19]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50) # lr = 0.1, est = 100

gbc.fit(X_train,Y_train)

YG1_pred = gbc.predict(X_test)
pd.crosstab(YG1_pred, YG1_pred)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YG1_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YG1_pred)
print(cm)
ac = accuracy_score(Y_test,YG1_pred)
print(ac)


# In[20]:


# Ada Boost Classifier
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
from sklearn.ensemble import AdaBoostClassifier
adbc = AdaBoostClassifier(base_estimator=ET,n_estimators=50) 
adbc.fit(X_train,Y_train)
YA_pred = adbc.predict(X_test)
pd.crosstab(YA_pred, YA_pred)

#YAT_pred = adbc.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,YAT_pred)
#print(cm)
#ac = accuracy_score(Y_train,YAT_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YA_pred = adbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YA_pred)
print(cm)
ac = accuracy_score(Y_test,YA_pred)
print(ac)


# # Tree graph

# In[22]:


from sklearn import tree
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
ET.fit(X_train, Y_train) 
tree.plot_tree(ET)  


# In[23]:


import matplotlib.pyplot as plt
fn=['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
cn=['Medium', 'High', 'Low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=600)
tree.plot_tree(ET,
               feature_names = fn, 
               class_names=cn,
               filled = True); 


# # Gini

# In[21]:


#Gini

# SPLITING DATA IN TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.33, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))


# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
DT.fit(X_train,Y_train)

# Predicted Y, Confusion Matrix and Accuracy score for TRAINNING data
#Y1_pred = DT.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,Y1_pred)
#print(cm)
#ac = accuracy_score(Y_train,Y1_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
Y_pred = DT.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
ac = accuracy_score(Y_test,Y_pred)
print(ac)


# In[22]:


from sklearn.metrics import classification_report  
print(classification_report(Y_pred, Y_test)) 


# In[23]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50) # lr = 0.1, est = 100

gbc.fit(X_train,Y_train)

YG1_pred = gbc.predict(X_test)
pd.crosstab(YG1_pred, YG1_pred)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YG1_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YG1_pred)
print(cm)
ac = accuracy_score(Y_test,YG1_pred)
print(ac)


# In[24]:


from sklearn.metrics import classification_report  
print(classification_report(YG1_pred, Y_test)) 


# In[25]:


# Ada Boost Classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.ensemble import AdaBoostClassifier
adbc = AdaBoostClassifier(base_estimator=DT,n_estimators=50) 
adbc.fit(X_train,Y_train)
YA_pred = adbc.predict(X_test)
pd.crosstab(YA_pred, YA_pred)

#YAT_pred = adbc.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,YAT_pred)
#print(cm)
#ac = accuracy_score(Y_train,YAT_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YA_pred = adbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YA_pred)
print(cm)
ac = accuracy_score(Y_test,YA_pred)
print(ac)


# In[26]:


from sklearn.metrics import classification_report  
print(classification_report(YA_pred, Y_test)) 


# # Using KFold

# In[27]:


# Using KFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 7, shuffle = True)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.model_selection import cross_val_score
results = cross_val_score(DT, X, Y, cv=kfold)
print(results)
print(np.mean(abs(results)))
print(pd.crosstab(Y,Y))

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
from sklearn.model_selection import KFold, cross_val_predict
Y1_pred = cross_val_predict(DT, X, Y, cv=kfold)
print(pd.crosstab(Y1_pred,Y1_pred))
print(pd.crosstab(Y,Y1_pred))


# In[28]:


from sklearn.metrics import classification_report  
print(classification_report(Y1_pred, Y)) 


# In[29]:


from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
DT.fit(X_train,Y_train)
tree.plot_tree(DT)


# In[30]:


import matplotlib.pyplot as plt
fn=['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
cn=['Medium', 'High', 'Low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=600)
tree.plot_tree(DT,
               feature_names = fn, 
               class_names=cn,
               filled = True);           


# # 2a. Dividing Sales in Two Categories (Low, High)
#     # Spliting variables in Train and Test

# In[31]:


# converting Sales value in catagorical Data
Sales_C = []
for Sales in df["Sales"]:
    if Sales <=7:
        Sales_C.append("Low")
    else:
        Sales_C.append("High")
df["Sales_C"]=Sales_C
df2 = df
df2.head()


# In[32]:


# ENcoding of Catagorical columns
LE = LabelEncoder()
df2["ShelveLoc"] = LE.fit_transform(df2["ShelveLoc"])
df2["Urban"] = LE.fit_transform(df2["Urban"])
df2["US"] = LE.fit_transform(df2["US"])
print(df2.head())
print(df2.shape)


# In[33]:


sns.countplot(df2.Sales_C)


# In[34]:


# X and Y variables
X2 = df2.iloc[:,1:11]
print(list(X2))
Y2 = df2.iloc[:,11]
print(pd.crosstab(Y2,Y2))


# In[35]:


# SPLITING DATA IN TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size = 0.20, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))


# In[36]:


#Entropy 

# SPLITING DATA IN TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))

# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
ET.fit(X_train, Y_train) 

# Predicted Y, Confusion Matrix and Accuracy score for TRAINNING data
#Y1_pred = ET.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,Y1_pred)
#print(cm)
#ac = accuracy_score(Y_train,Y1_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
Y_pred = ET.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
ac = accuracy_score(Y_test,Y_pred)
print(ac)


# In[37]:


# Using KFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 7, shuffle = True)
from sklearn.tree import DecisionTreeClassifier
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
from sklearn.model_selection import cross_val_score
results = cross_val_score(ET, X2, Y2, cv=kfold)
print(results)
print(np.mean(abs(results)))
# Predicted Y, Confusion Matrix and Accuracy score for TEST data
from sklearn.model_selection import KFold, cross_val_predict
Y1_pred = cross_val_predict(ET, X2, Y2, cv=kfold)
print(pd.crosstab(Y,Y))
print(pd.crosstab(Y1_pred,Y1_pred))
print(pd.crosstab(Y,Y1_pred))


# In[38]:


from sklearn.metrics import classification_report  
print(classification_report(Y1_pred, Y)) 


# In[39]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50) # lr = 0.1, est = 100

gbc.fit(X_train,Y_train)

YG1_pred = gbc.predict(X_test)
pd.crosstab(YG1_pred, YG1_pred)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YG1_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YG1_pred)
print(cm)
ac = accuracy_score(Y_test,YG1_pred)
print(ac)


# In[40]:


# Ada Boost Classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.ensemble import AdaBoostClassifier
adbc = AdaBoostClassifier(base_estimator=ET,n_estimators=50) 
adbc.fit(X_train,Y_train)
YA_pred = adbc.predict(X_test)
pd.crosstab(YA_pred, YA_pred)

#YAT_pred = adbc.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,YAT_pred)
#print(cm)
#ac = accuracy_score(Y_train,YAT_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YA_pred = adbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YA_pred)
print(cm)
ac = accuracy_score(Y_test,YA_pred)
print(ac)


# In[41]:


from sklearn import tree
ET = DecisionTreeClassifier(criterion='entropy', max_depth = 3)
ET.fit(X_train, Y_train) 
tree.plot_tree(ET)  


# In[42]:


import matplotlib.pyplot as plt
fn=['CompPrice', 'Income', 'Advertising', 'Population', 'Price',
       'ShelveLoc', 'Age', 'Education', 'Urban', 'US']
cn=['High', 'Low']
fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (6,6), dpi=600)
tree.plot_tree(ET,
               feature_names = fn, 
               class_names=cn,
               filled = True); 


# # GINI

# In[43]:


#Gini

# SPLITING DATA IN TRAIN AND TEST
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X2, Y2, test_size = 0.33, random_state = 40)
print(X_train.shape)
print(Y_train.shape)
print(X_test.shape)
print(Y_test.shape)
print(pd.crosstab(Y_train,Y_train))
print(pd.crosstab(Y_test,Y_test))


# DECISION TREE CLASSIFIER
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
DT.fit(X_train,Y_train)

# Predicted Y, Confusion Matrix and Accuracy score for TRAINNING data
Y1_pred = DT.predict(X_train)
from sklearn.metrics import confusion_matrix,accuracy_score
cm1 = confusion_matrix(Y_train,Y1_pred)
print(cm1)
ac1 = accuracy_score(Y_train,Y1_pred)
print(ac1)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
Y_pred = DT.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
ac = accuracy_score(Y_test,Y_pred)
print(ac)


# In[44]:


from sklearn.metrics import classification_report  
print(classification_report(Y_pred, Y_test)) 


# In[45]:


#Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(learning_rate=0.1,n_estimators=50) # lr = 0.1, est = 100

gbc.fit(X_train,Y_train)

YG1_pred = gbc.predict(X_test)
pd.crosstab(YG1_pred, YG1_pred)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YG1_pred = gbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YG1_pred)
print(cm)
ac = accuracy_score(Y_test,YG1_pred)
print(ac)


# In[46]:


from sklearn.metrics import classification_report  
print(classification_report(YG1_pred, Y_test))


# In[47]:


# Ada Boost Classifier
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.ensemble import AdaBoostClassifier
adbc = AdaBoostClassifier(base_estimator=DT,n_estimators=50) 
adbc.fit(X_train,Y_train)
YA_pred = adbc.predict(X_test)
pd.crosstab(YA_pred, YA_pred)

#YAT_pred = adbc.predict(X_train)
#from sklearn.metrics import confusion_matrix,accuracy_score
#cm = confusion_matrix(Y_train,YAT_pred)
#print(cm)
#ac = accuracy_score(Y_train,YAT_pred)
#print(ac)

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
YA_pred = adbc.predict(X_test)
from sklearn.metrics import confusion_matrix,accuracy_score
cm = confusion_matrix(Y_test,YA_pred)
print(cm)
ac = accuracy_score(Y_test,YA_pred)
print(ac)


# In[48]:


from sklearn.metrics import classification_report  
print(classification_report(YA_pred, Y_test)) 


# In[ ]:





# In[ ]:


# 2b. Dividing Sales in Two Categories (Low, High)
    # Using KFold


# In[49]:


# Using KFold
from sklearn.model_selection import KFold
kfold = KFold(n_splits=10, random_state = 7, shuffle = True)
from sklearn.tree import DecisionTreeClassifier
DT = DecisionTreeClassifier(criterion='gini', max_depth = 3)
from sklearn.model_selection import cross_val_score
results = cross_val_score(DT, X2, Y2, cv=kfold)
print(results)
print(np.mean(abs(results)))
print(pd.crosstab(Y,Y))

# Predicted Y, Confusion Matrix and Accuracy score for TEST data
from sklearn.model_selection import KFold, cross_val_predict
Y1_pred = cross_val_predict(DT, X2, Y2, cv=kfold)
print(pd.crosstab(Y1_pred,Y1_pred))
print(pd.crosstab(Y,Y1_pred))


# In[50]:


from sklearn.metrics import classification_report  
print(classification_report(Y1_pred, Y)) 

