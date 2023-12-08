# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 18:31:33 2021

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
#bring in the data
filename="Textdataset1.csv"
smalldata=pd.read_csv(filename)
print(type(smalldata))
print(smalldata)

# KMEANS
# Use k-means clustering on the data.
# Create clusters 
k = 3
## Sklearn required you to instantiate first
kmeans = KMeans(n_clusters=k)
kmeans.fit(smalldata)   ## run kmeans
labels = kmeans.labels_
print(labels)
centroids = kmeans.cluster_centers_
print(centroids)
prediction = kmeans.predict(smalldata)
print(prediction)

k = 4
## Sklearn required you to instantiate first
kmeans = KMeans(n_clusters=k)
kmeans.fit(smalldata)   ## run kmeans
labels = kmeans.labels_
print(labels)
centroids = kmeans.cluster_centers_
print(centroids)
prediction = kmeans.predict(smalldata)
print(prediction)

k = 5
## Sklearn required you to instantiate first
kmeans = KMeans(n_clusters=k)
kmeans.fit(smalldata)   ## run kmeans
labels = kmeans.labels_
print(labels)
centroids = kmeans.cluster_centers_
print(centroids)
prediction = kmeans.predict(smalldata)
print(prediction)
##         Look at best values for k
##
###################################################
SS_dist = []
values_for_k=range(2,9)
#print(values_for_k)
for k_val in values_for_k:
    print(k_val)
    k_means = KMeans(n_clusters=k_val)
    model = k_means.fit(smalldata)
    SS_dist.append(k_means.inertia_)
    
print(SS_dist)
print(values_for_k)
plt.plot(values_for_k, SS_dist, 'bx-')
plt.xlabel('value')
plt.ylabel('Sum of squared distances')
plt.title('Elbow method for optimal k Choice')
plt.show()
####
# Look at Silhouette
##########################
Sih=[]
Cal=[]
k_range=range(2,9)
for k in k_range:
    k_means_n = KMeans(n_clusters=k)
    model = k_means_n.fit(smalldata)
    Pred = k_means_n.predict(smalldata)
    labels_n = k_means_n.labels_
    R1=metrics.silhouette_score(smalldata, labels_n, metric = 'euclidean')
    R2=metrics.calinski_harabasz_score(smalldata, labels_n)
    Sih.append(R1)
    Cal.append(R2)
print(Sih) ## higher is better
print(Cal) ## higher is better
fig1, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(k_range,Sih)
ax1.set_title("Silhouette")
ax1.set_xlabel("")
ax2.plot(k_range,Cal)
ax2.set_title("Calinski_Harabasz_Score")
ax2.set_xlabel("k values")
####################################################


