# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 13:40:10 2021

@author: Jieyi Sun
"""

import pandas as pd
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB,CategoricalNB

data=pd.read_csv(r'E:\GU\501\NB_SVM\Textdataset.csv')

#Scale the data
Scaler=StandardScaler()
data.iloc[:,:-1]=Scaler.fit_transform(data.iloc[:,:-1])

#train-test split
X_train, X_test, y_train, y_test=train_test_split(data.iloc[:,:-1],data.LABEL,stratify=data.LABEL,test_size=0.2,random_state=32)
print(y_train.value_counts())
print(y_test.value_counts())

#Tune var_smoothing
Accuracy=[]
for var_smoothing in [1e-1,1e-3,1e-5,1e-7,1e-9,1e-10,1e-11]:
    model=GaussianNB(var_smoothing=var_smoothing)
    model.fit(X_train,y_train)
    y_pred_test=model.predict(X_test)
    Accuracy.append(accuracy_score(y_test,y_pred_test))
    
plt.plot(range(7),Accuracy)
plt.xlabel('Var_smoothing')
plt.ylabel('Accuracy')
plt.xticks(range(7),labels=['1e-1','1e-3','1e-5','1e-7','1e-9','1e-10','1e-11'],rotation=90)
plt.show()
#CrossValidation
model=GaussianNB()
cv_res=cross_val_score(model,X_train,y_train,scoring='accuracy',cv=5)
print('CV accuracy:',np.mean(cv_res))
#CV accuracy: 0.375

model=GaussianNB()
model.fit(X_train,y_train)
y_pred_train=model.predict(X_train)
print("Train"+classification_report(y_train,y_pred_train))

y_pred_test=model.predict(X_test)
print("Test"+classification_report(y_test,y_pred_test))

labels=y_train.unique()
cm=pd.DataFrame(confusion_matrix(y_train,y_pred_train,labels=labels),index=labels,columns=labels)
sns.heatmap(cm,vmin=0,vmax=10,annot=True,cmap='Blues',cbar=False)
plt.title('Training Confusion Matrix')
plt.show()

cm=pd.DataFrame(confusion_matrix(y_test,y_pred_test,labels=labels),index=labels,columns=labels)
sns.heatmap(cm,vmin=0,vmax=10,annot=True,cmap='Oranges',cbar=False)
plt.title('Test Confusion Matrix')
plt.show()

plt.bar(x=y_train.unique(),height=model.class_prior_,color="orange")
plt.axhline(y=model.class_prior_[0],color='red',linestyle='dashed')
plt.text(-1.25,model.class_prior_[0],round(model.class_prior_[0],2))
plt.title('Prior_Prob of Labels')
plt.show()