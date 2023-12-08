# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 02:54:01 2021

@author: Jieyi Sun
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

data=pd.read_csv(r'E:\GU\501\NB_SVM\Textdataset.csv')
#Check the labels
print(data.LABEL.value_counts())

#Scaling
standardscaler = StandardScaler()
data.iloc[:,:-1]=standardscaler.fit_transform(data.iloc[:,:-1])

#PCA
pca=PCA(n_components=0.95,svd_solver='full')
data_pca=pca.fit_transform(data.iloc[:,:-1])
data_pca=pd.DataFrame(data_pca,columns=['P'+str(i) for i in range(data_pca.shape[1])])
data_pca['LABEL']=data['LABEL']
data=data_pca

#train-test split
X_train, X_test, y_train, y_test=train_test_split(data.iloc[:,:-1],data.LABEL,stratify=data.LABEL,test_size=0.2,random_state=123)
print(y_train.value_counts())
print(y_test.value_counts())
#### Linear Kernel
Accuracy=[]
for i in np.linspace(0.1,1.5,14):
    linear_model=SVC(kernel='linear',probability=True,random_state=42,C=i)
    linear_model.fit(X_train,y_train)
    linear_y_pred=linear_model.predict(X_test)
    Accuracy.append(accuracy_score(linear_y_pred,y_test))

plt.plot(np.linspace(0,1.5,14),Accuracy,'b')
plt.title('C of Linear Kernel-Accuracy')
plt.xlabel('C of Linear Kernel')
plt.ylabel('Accuracy')
plt.show()
##from the graph we can see that accuracy is the same for all C values 
linear_model=SVC(kernel='linear',probability=True,random_state=123,C=0.2)
linear_model.fit(X_train,y_train)
linear_y_pred=linear_model.predict(X_test)
print(classification_report(y_test,linear_y_pred))

labels=y_train.unique()
linear_confusion_matrix=pd.DataFrame(confusion_matrix(y_test,linear_y_pred,labels=labels),index=labels,columns=labels)
sns.heatmap(linear_confusion_matrix,vmin=0,vmax=10,annot=True,cmap='Greens',cbar=False)
plt.title('Linear Kernel SVM')
plt.show()

#### Poly Kernel
Accuracy=[]
for i in range(6):
    poly_model=SVC(kernel='poly',random_state=42,degree=i)
    poly_model.fit(X_train,y_train)
    poly_y_pred=poly_model.predict(X_test)
    Accuracy.append(accuracy_score(poly_y_pred,y_test))

plt.plot(range(6),Accuracy,'b')
plt.title('Degree of PolyNominal Kernel-Accuracy')
plt.xlabel('Degree of PolyNominal Kernel')
plt.ylabel('Accuracy')
plt.show()
##From the graph we can see that accuracy is the highest when degree of PolyNominal Kernel is 1
poly_model=SVC(kernel='poly',random_state=42,degree=1)
poly_model.fit(X_train,y_train)
poly_y_pred=poly_model.predict(X_test)
print(classification_report(y_test,poly_y_pred))

labels=y_train.unique()
poly_confusion_matrix=pd.DataFrame(confusion_matrix(y_test,poly_y_pred,labels=labels),index=labels,columns=labels)
sns.heatmap(poly_confusion_matrix,vmin=0,vmax=10,annot=True,cmap='Greens',cbar=False)
plt.title('POLY Kernel SVM')
plt.show()

#### rbf Kernel
Accuracy=[]
Gamma=np.linspace(0.01,4,20)
for i in Gamma:
    rbf_model=SVC(kernel='rbf',probability=True,random_state=42,gamma=i)
    rbf_model.fit(X_train,y_train)
    rbf_y_pred=rbf_model.predict(X_test)
    Accuracy.append(accuracy_score(rbf_y_pred,y_test))

plt.plot(Gamma,Accuracy,'b')
plt.title('Gamma of RBF Kernel-Accuracy')
plt.xlabel('Gamma of RNF Kernel')
plt.ylabel('Accuracy')
plt.show()
##from the graph we can see that accuracy is the highest when degree is 1.3
rbf_model=SVC(kernel='rbf',probability=True,random_state=42,gamma=1.3)
rbf_model.fit(X_train,y_train)
rbf_y_pred=rbf_model.predict(X_test)
print(classification_report(y_test,rbf_y_pred))

labels=y_train.unique()
rbf_confusion_matrix=pd.DataFrame(confusion_matrix(y_test,rbf_y_pred,labels=labels),index=labels,columns=labels)
sns.heatmap(rbf_confusion_matrix,vmin=0,vmax=10,annot=True,cmap='Greens',cbar=False)
plt.title('RBF Kernel SVM')
plt.show()








