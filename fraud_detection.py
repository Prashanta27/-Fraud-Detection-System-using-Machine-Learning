#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import seaborn as sns
from narwhals.selectors import categorical


# In[2]:


import warnings
warnings.filterwarnings("ignore")
sns.set(style="whitegrid")


# In[3]:


dataset = pd.read_csv("fraud detection.csv")
dataset.head(3)


# In[4]:


dataset.isnull().sum()


# In[5]:


dataset.describe()


# In[6]:


dataset["isFraud"].value_counts()


# In[7]:


dataset["isFlaggedFraud"].value_counts()


# In[8]:


dataset.shape[0]


# In[9]:


dataset.isnull().sum().sum()


# In[10]:


round((dataset["isFraud"].value_counts()[1]/dataset.shape[0]) * 100,2)


# In[11]:


plt.figure(figsize = (4,3))
dataset["type"].value_counts().plot(kind="bar",title="Transaction Type", color = "skyblue")
plt.xlabel("Transaction Type")
plt.ylabel("count")
plt.show()


# In[12]:


fraud_by_tupe = dataset.groupby("type")["isFraud"].mean().sort_values(ascending=False)
fraud_by_tupe.plot(kind="bar", title = "Fraud Rate By Type", color = "salmon")
plt.ylabel("Fraud Rate")
plt.show()


# In[13]:


dataset["amount"].describe().astype(int)


# In[14]:


sns.histplot(np.log1p(dataset["amount"]), bins = 100, kde = True, color = "green")
plt.title("Transaction Amount Distribution (log scale)")
plt.xlabel("log(Amount + 1)")
plt.show()


# In[15]:


sns.boxplot(data = dataset[dataset["amount"] < 50000], x = "isFraud", y = "amount")
plt.title("Amount vs isFraud (Filtered under 50k)")
plt.show()


# In[16]:


dataset["BalanceDiffOrig"] = dataset["oldbalanceOrg"] - dataset["newbalanceOrig"]  #new column create
dataset["BalanceDiffDest"] = dataset["newbalanceDest"] - dataset["oldbalanceDest"] #new column create


# In[17]:


(dataset["BalanceDiffOrig"] < 0).sum()


# In[18]:


(dataset["BalanceDiffDest"] < 0).sum()


# In[19]:


dataset.head(2)


# In[20]:


fraud_per_step = dataset[dataset["isFraud"] == 1]["step"].value_counts().sort_index()
plt.plot(fraud_per_step.index,fraud_per_step.values, label = "Fraud per step")
plt.xlabel("Step (Time)")
plt.ylabel("Number Of Frauds")
plt.title("Frauds Over Time")
plt.grid(True)
plt.show()


# In[21]:


dataset.drop(columns="step", inplace=True)


# In[22]:


dataset.head(2)


# In[23]:


top_sender = dataset["nameOrig"].value_counts().head(10)


# In[24]:


top_sender


# In[25]:


top_receivers = dataset["nameDest"].value_counts().head()


# In[26]:


top_receivers


# In[27]:


fraud_users = dataset[dataset["isFraud"] == 1] ["nameOrig"].value_counts()


# In[28]:


fraud_users.shape


# In[29]:


fraud_type = dataset[dataset["type"].isin(["TRANSFER","CASH_OUT"])]


# In[30]:


fraud_type["type"].value_counts()


# In[31]:


plt.figure(figsize=(4,3))
sns.countplot(data = fraud_type, x= "type", hue = "isFraud")
plt.title("Fraud Distribution In Transfer & Cash_Out")
plt.show()


# In[32]:


corr = dataset.corr(numeric_only=True)


# In[33]:


corr


# In[34]:


sns.heatmap(corr,annot=True,cmap = "coolwarm", fmt= ".2f")
plt.title("Correlation Matrix")
plt.show()


# In[35]:


zero_after_transfer = dataset[
    (dataset["oldbalanceOrg"] > 0) &
    (dataset["newbalanceOrig"] == 0) &
    (dataset["type"].isin(["TRANSFER","CASH_OUT"]))
]


# In[36]:


len(zero_after_transfer)


# In[37]:


zero_after_transfer.head(3)


# In[38]:


dataset["isFraud"].value_counts()


# In[39]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


# In[40]:


dataset.head(3)


# In[41]:


dataset_model = dataset.drop(["nameOrig","nameDest","isFlaggedFraud"], axis = 1)
dataset_model


# In[42]:


categorical = ["type"]
numarical = ["amount","oldbalanceOrg","newbalanceOrig","oldbalanceDest","newbalanceDest"]


# In[43]:


y = dataset_model["isFraud"]
x = dataset_model.drop("isFraud",axis = 1)


# In[44]:


x_train,x_test,y_train,y_test = train_test_split(x,y, test_size=0.3, stratify=y, random_state=42)


# In[45]:


#Data Preprocessing

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numarical),
        ("cat", OneHotEncoder(drop="first"), categorical)
    ],
    remainder= "drop"

)


# In[46]:


pipeline = Pipeline([
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])


# In[47]:


# Now Train the model
pipeline.fit(x_train,y_train)


# In[48]:


y_pred = pipeline.predict(x_test)


# In[49]:


y_pred


# In[50]:


print(classification_report(y_test, y_pred)) #from metrix


# In[51]:


confusion_matrix(y_test, y_pred) #from metrix


# In[52]:


accuracy_score(y_test, y_pred) #from metrics


# In[53]:


pipeline.score(x_test,y_test)


# In[54]:


import pickle


# In[55]:


pickle.dump(pipeline,open("Fraud_detection.pkl", "wb"))


# In[ ]:




