# Importing required libraries
import numpy as np
from keras import optimizers
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from tensorflow import keras
from tensorflow.keras.layers import Dense
from sklearn.metrics import accuracy_score, classification_report,recall_score,roc_auc_score,confusion_matrix,precision_score

# Loading the data
try:
    data = pd.read_csv('Admission_Predict.csv')
except FileNotFoundError:
    print("File not found")


# Renaming column to remove extra space
print(data.info())
data = data.rename(columns = {'Chance of Admit ':' Chance of Admit'})


# Checking for missing values
print(data.isnull().any())

# In[21]:


print(data.describe())


# In[22]:


sns.distplot(data['GRE Score'])


# In[23]:


sns.pairplot(data=data,hue='Research',markers=["^","v"],palette='inferno')


# In[24]:


sns.scatterplot(x='University Rating',y='CGPA',data=data,color='Red', s=100)


# In[42]:


category = ['GRE Score','TOEFL Score','University Rating','SOP','LOR','CGPA','Research','ChanceofAdmit']
color = ['yellowgreen','gold','lightskyblue','pink','red','purple','orange','gray']
start = True
for i in np.arange(4):
    fig = plt.figure(figsize=(14,8))
    plt.subplot2grid((4,2),(i,0))
    data[category[2*i]].hist(color=color[2*i],bins=10)
    plt.title(category[2*i])
    plt.subplot2grid((4,2),(i,1))
    data[category[2*i+1]].hist(color=color[2*i+1],bins=10)
    plt.title(category[2*i+1])
    
plt.subplots_adjust(hspace = 0.7, wspace= 0.2)
plt.show()


# In[28]:
sc=MinMaxScaler()
x=sc.fit_transform(data.iloc[:,0:7].values)
print(x)


# In[29]:


x=data.iloc[:,0:7].values
print(x)


# In[30]:


y=data.iloc[:,7:].values
print(y)


# In[31]:



x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.30,random_state=101)


# In[32]:


y_train=(y_train>0.5)
print(y_train)


# In[33]:


y_test=(y_test>0.5)


# In[37]:
cls = LogisticRegression(random_state =0)

lr = cls.fit(x_train, y_train[:, 0])

# In[ ]:


y_pred = lr.predict(x_test)
print(y_pred)

# In[39]:


model=keras.Sequential()
model.add(Dense(7,activation = 'relu',input_dim=7))
model.add(Dense(7,activation='relu'))
model.add(Dense(1,activation='linear'))

model.summary()


# In[40]:


model.compile(optimizer=optimizers.Adam(lr=0.001),loss='mse')

model.fit(x_train, y_train, batch_size=20, epochs=100)

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

from sklearn.metrics import accuracy_score

train_predictions = model.predict(x_train)

print(train_predictions)


# In[40]:


train_acc = model.evaluate(x_train, y_train, verbose=0)[1]


# In[41]:


test_acc = model.evaluate(x_test, y_test, verbose=0)[1]


# In[43]:


pred=model.predict(x_test)
pred=(pred>0.5)
pred


# In[44]:


y_pred_prob = model.predict(x_test)

# convert probabilities to class labels
y_pred = np.argmax(y_pred_prob, axis=1)

from sklearn.metrics import accuracy_score,recall_score,roc_auc_score,confusion_matrix
print("\nAccuracy score:%f" %(accuracy_score(np.argmax(y_test, axis=1),y_pred) * 100 ))
print("Recall score: %f" %(recall_score(np.argmax(y_test, axis=1),y_pred) * 100))
print("ROC score: %f\n" %(roc_auc_score(np.argmax(y_test, axis=1),y_pred) * 100))

print(confusion_matrix(np.argmax(y_test, axis=1),y_pred))

# In[21]:
model.save('Admission.h5')
