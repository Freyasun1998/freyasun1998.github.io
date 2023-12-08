# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 02:31:20 2021

@author: Jieyi Sun
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from sklearn import datasets
from sklearn import preprocessing
import pylab as pl
from sklearn import decomposition
import seaborn as sns
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN

filename="Textdataset1.csv"
smalldata=pd.read_csv(filename)
print(type(smalldata))
print(smalldata)

# It is often best to normalize the data 
## before applying the fit method
## There are many normalization options
## This is an example of using the z score
smalldata_normalized=(smalldata - smalldata.mean()) / smalldata.std()
print(smalldata_normalized)
print(smalldata_normalized.shape[0])   ## num rows
print(smalldata_normalized.shape[1])   ## num cols
NumCols=smalldata_normalized.shape[1]
## Instantiated my own copy of PCA
My_pca = PCA(n_components=2)  ## I want the two prin columns
## Transpose it
smalldata_normalized=np.transpose(smalldata_normalized)
My_pca.fit(smalldata_normalized)
print(My_pca)
print(My_pca.components_.T)
KnownLabels=["causes", "diagnosis", "lifestyle", "medications", "prevention"]

# Reformat and view results
Comps = pd.DataFrame(My_pca.components_.T,
                        columns=['PC%s' % _ for _ in range(2)],
                        index=smalldata_normalized.columns
                        )
print(Comps)
print(Comps.iloc[:,0])

## Look at 2D PCA clusters
############################################
plt.figure(figsize=(12,12))
plt.scatter(Comps.iloc[:,0], Comps.iloc[:,1], s=100, color="green")
plt.xlabel("PC 1")
plt.ylabel("PC 2")
plt.title("Scatter Plot Clusters PC 1 and 2",fontsize=15)
for i in range(1,10):
    label="causes"
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))
for i in range(11,20):
    label="diagnosis"
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))
for i in range(21,30):
    label="lifestyle"
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))
for i in range(31,40):
    label="medications"
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))
for i in range(41,50):
    label="prevention"
    plt.annotate(label, (Comps.iloc[i,0], Comps.iloc[i,1]))
plt.savefig("PCA.png")
plt.show()
##         DBSCAN
##
###############################################

MyDBSCAN = DBSCAN(eps=0.0001, min_samples=1)
y_pred=MyDBSCAN.fit_predict(My_pca.components_.T)
plt.scatter(My_pca.components_.T[:, 0], My_pca.components_.T[:, 1], c=y_pred)
plt.title("DBSCAN for reddit",fontsize=15)
plt.savefig("dbscan.png")
plt.show()
## eps:
    ## The maximum distance between two samples for 
    ##one to be considered as in the neighborhood of the other.
##  Hierarchical 
##
#########################################

MyHC = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
FIT=MyHC.fit(smalldata)
HC_labels = MyHC.labels_
print(HC_labels)

plt.figure(figsize =(12, 12))
plt.title('Hierarchical Clustering')
dendro = hc.dendrogram((hc.linkage(smalldata, method ='ward')))

