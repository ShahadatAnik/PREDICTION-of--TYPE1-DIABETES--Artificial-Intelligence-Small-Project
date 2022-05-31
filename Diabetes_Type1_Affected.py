#!/usr/bin/env python
# coding: utf-8

# # PREDICTION OF ‘TYPE1 DIABETES’

# In[1]:


import pandas as pd
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('Diabetes_Type1.csv')


# In[3]:


data.head(2)


# In[4]:


SexColumnDummy = pd.get_dummies(data['Sex'])
data = pd.concat((data,SexColumnDummy), axis=1)
data = data.drop(['Sex'], axis=1)


# In[5]:


data.head(2)


# In[6]:


AgeColumnDummy = pd.get_dummies(data['Age'])
data = pd.concat((data,AgeColumnDummy), axis=1)
data = data.drop(['Age'], axis=1)


# In[7]:


data = data.drop(['Area of Residence '], axis=1)


# In[8]:


HbA1cofResidenceColumnDummy = pd.get_dummies(data['HbA1c'])
data = pd.concat((data,HbA1cofResidenceColumnDummy), axis=1)
data = data.drop(['HbA1c'], axis=1)
data = data.drop(['Duration of disease'], axis=1)


# In[9]:


OtherdieaseColumnDummy = pd.get_dummies(data['Other diease'])
data = pd.concat((data,OtherdieaseColumnDummy), axis=1)
data = data.drop(['Other diease'], axis=1)


# In[10]:


AdequateNutritionColumnDummy = pd.get_dummies(data['Adequate Nutrition '])
data = pd.concat((data,AdequateNutritionColumnDummy), axis=1)
data = data.drop(['Adequate Nutrition '], axis=1)


# In[11]:


data = data.drop(['Education of Mother'], axis=1)


# In[12]:


StandardizedgrowthrateininfancyColumnDummy = pd.get_dummies(data['Standardized growth-rate in infancy'])
data = pd.concat((data,StandardizedgrowthrateininfancyColumnDummy), axis=1)
data = data.drop(['Standardized growth-rate in infancy'], axis=1)


# In[13]:


StandardizedbirthweightColumnDummy = pd.get_dummies(data['Standardized birth weight'])
data = pd.concat((data,StandardizedbirthweightColumnDummy), axis=1)
data = data.drop(['Standardized birth weight'], axis=1)


# In[14]:


AutoantibodiesColumnDummy = pd.get_dummies(data['Autoantibodies'])
data = pd.concat((data,AutoantibodiesColumnDummy), axis=1)
data = data.drop(['Autoantibodies'], axis=1)


# In[15]:


ImpairedglucosemetabolismColumnDummy = pd.get_dummies(data['Impaired glucose metabolism '])
data = pd.concat((data,SexColumnDummy), axis=1)
data = data.drop(['Impaired glucose metabolism '], axis=1)


# In[16]:


InsulintakenColumnDummy = pd.get_dummies(data['Insulin taken'])
data = pd.concat((data,InsulintakenColumnDummy), axis=1)
data = data.drop(['Insulin taken'], axis=1)


# In[17]:


data = data.drop(['How Taken'], axis=1)


# In[18]:


FamilyHistoryaffectedinType1DiabetesColumnDummy = pd.get_dummies(data['Family History affected in Type 1 Diabetes'])
data = pd.concat((data,FamilyHistoryaffectedinType1DiabetesColumnDummy), axis=1)
data = data.drop(['Family History affected in Type 1 Diabetes'], axis=1)


# In[19]:


FamilyHistoryaffectedinType2DiabetesColumnDummy = pd.get_dummies(data['Family History affected in Type 2 Diabetes'])
data = pd.concat((data,FamilyHistoryaffectedinType2DiabetesColumnDummy), axis=1)
data = data.drop(['Family History affected in Type 2 Diabetes'], axis=1)


# In[20]:


HypoglycemisColumnDummy = pd.get_dummies(data['Hypoglycemis'])
data = pd.concat((data,HypoglycemisColumnDummy), axis=1)
data = data.drop(['Hypoglycemis'], axis=1)


# In[21]:


pancreaticdiseaseaffectedinchildColumnDummy = pd.get_dummies(data['pancreatic disease affected in child '])
data = pd.concat((data,SexColumnDummy), axis=1)
data = data.drop(['pancreatic disease affected in child '], axis=1)


# In[22]:


data.head(2)


# In[23]:


features = data.columns


# In[24]:


features = [x for x in features if x != 'Affected']


# In[25]:


train, test = train_test_split(data, test_size = 0.25)
print(len(data))
print(len(train))
print(len(test))


# # DECISION TREE

# In[26]:


dt = DecisionTreeClassifier(min_samples_split = 100, criterion='entropy')


# In[27]:


x_train = train[features]
y_train = train["Affected"]

x_test = test[features]
y_test = test["Affected"]


# In[28]:


dt = dt.fit(x_train, y_train)


# In[29]:


y_pred = dt.predict(x_test)


# ### ACCURACY

# In[30]:


score = accuracy_score(y_test, y_pred)*100
print("Accuracy using desicion Tree: ", round(score, 2), "%" )


# ### CONFUSION MATRIX

# In[31]:


plot_confusion_matrix(dt, x_test, y_test,cmap='PuBuGn',values_format='.1f')  
plt.title("Confusion matrix by Decision Tree")
plt.show()


# ### CLASSIFICATION REPORT

# In[32]:


print(classification_report(y_test,y_pred))


# ### TREE

# In[33]:


get_ipython().system('pip install graphviz')


# In[34]:


plt.figure(figsize=(20,20))
clf = DecisionTreeClassifier(max_depth = 5).fit(x_train, y_train)
plot_tree(clf, filled=True, fontsize=9)
plt.show()


# # RANDOM FOREST

# In[35]:


rf = RandomForestClassifier(n_estimators=100,max_depth=9)


# In[36]:


rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)


# ### ACCURACY

# In[37]:


score1 = accuracy_score(y_test, y_pred_rf)*100
print("Accuracy using random forest:",round(score1, 2), "%")


# ### CONFUSION MATRIX

# In[38]:


plot_confusion_matrix(rf, x_test, y_test,cmap='PuBuGn',values_format='.1f')  
plt.title("Confusion matrix by Random Forest")
plt.show()


# ### CLASSIFICATION REPORT

# In[39]:


print(classification_report(y_test,y_pred_rf))


# # Naive Bayes

# In[40]:


nb = GaussianNB()
nb.fit(x_train,y_train)
y_pred_nb = nb.predict(x_test)


# ### ACCURACY

# In[41]:


score2 = accuracy_score(y_test, y_pred_nb)*100
print("Accuracy using naive bayes:",round(score2, 2), "%")


# ### CONFUSION MATRIX

# In[42]:


plot_confusion_matrix(nb, x_test, y_test,cmap='PuBuGn',values_format='.1f')  
plt.title("Confusion matrix by Naïve Bayes")
plt.show()


# ### CLASSIFICATION REPORT

# In[43]:


print(classification_report(y_test,y_pred_nb))

