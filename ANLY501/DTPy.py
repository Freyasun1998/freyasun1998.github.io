# -*- coding: utf-8 -*-
"""
Created on Thu Nov 11 00:35:47 2021

@author: Jieyi Sun
"""

import numpy as np
import pandas as pd
import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
from nltk.stem import PorterStemmer
import os
import re   
from mpl_toolkits.mplot3d import Axes3D
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import seaborn as sns
import sys
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
from sklearn.model_selection import train_test_split
import random as rd
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn import tree
from sklearn.decomposition import LatentDirichletAllocation 
from sklearn import preprocessing
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
from mpl_toolkits.mplot3d import Axes3D
from scipy.cluster.hierarchy import ward, dendrogram
##STEP 1   Create Training and Testing Data
###############################################################
## Write the dataframe to csv so you can use it later if you wish
filename="Textdataset.csv"
Final_News_DF_Labeled=pd.read_csv(filename)
TrainDF, TestDF = train_test_split(Final_News_DF_Labeled, test_size=0.3)
print(TrainDF)
print(TestDF)
#################################################
## STEP 2: Separate LABELS
#################################################
## IMPORTANT - YOU CANNOT LEAVE LABELS ON 
## Save labels
### TEST ---------------------
TestLabels=TestDF["LABEL"]
print(TestLabels)
TestDF = TestDF.drop(["LABEL"], axis=1)
print(TestDF)
### TRAIN----------------------
TrainLabels=TrainDF["LABEL"]
print(TrainLabels)
## remove labels
TrainDF = TrainDF.drop(["LABEL"], axis=1)
TestDF.to_csv("TestDF.csv")
TrainDF.to_csv("TrainDF.csv")
##################################################
## STEP 3:  Run DT 1
##################################################
## Instantiate
MyDT1=DecisionTreeClassifier(criterion='gini', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

##
MyDT1.fit(TrainDF, TrainLabels)

##Visualization
feature_names1=TrainDF.columns
fig1=plt.figure(figsize=(40,30))
Tree_Object1=tree.plot_tree(MyDT1,
                            feature_names=feature_names1,
                            class_names='LABEL',
                            filled=True)
fig1.savefig("DT1.png")


from sklearn import metrics
from sklearn.metrics import classification_report
DT_pred1=MyDT1.predict(TestDF)

##Run DT 2
MyDT2=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='best',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

##
MyDT2.fit(TrainDF, TrainLabels)

##Visualization
feature_names2=TrainDF.columns
fig2=plt.figure(figsize=(40,30))
Tree_Object1=tree.plot_tree(MyDT2,
                            feature_names=feature_names2,
                            class_names='LABEL',
                            filled=True)
fig2.savefig("DT2.png")


from sklearn import metrics
from sklearn.metrics import classification_report
DT_pred2=MyDT2.predict(TestDF)

##Run DT 3
MyDT3=DecisionTreeClassifier(criterion='entropy', ##"entropy" or "gini"
                            splitter='random',  ## or "random" or "best"
                            max_depth=None, 
                            min_samples_split=2, 
                            min_samples_leaf=1, 
                            min_weight_fraction_leaf=0.0, 
                            max_features=None, 
                            random_state=None, 
                            max_leaf_nodes=None, 
                            min_impurity_decrease=0.0, 
                            min_impurity_split=None, 
                            class_weight=None)

##
MyDT3.fit(TrainDF, TrainLabels)

##Visualization
feature_names3=TrainDF.columns
fig1=plt.figure(figsize=(40,30))
Tree_Object1=tree.plot_tree(MyDT3,
                            feature_names=feature_names3,
                            class_names='LABEL',
                            filled=True)
fig1.savefig("DT3.png")


#confision matrix for DT3
print("Prediction\n")
DT_pred3=MyDT3.predict(TestDF)
print(DT_pred3)

##
    
bn_matrix3= confusion_matrix(TestLabels, DT_pred3)
print("\nThe confusion matrix is:")
print(bn_matrix3)

FeatureImp=MyDT3.feature_importances_   
indices3= np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices3[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices3[f], FeatureImp[indices3[f]]))
        print ("feature name: ", feature_names3[indices3[f]])

#confision matrix for DT2
print("Prediction\n")
bn_matrix2= confusion_matrix(TestLabels, DT_pred2)
print("\nThe confusion matrix is:")
print(bn_matrix2)

FeatureImp=MyDT2.feature_importances_   
indices2= np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices2[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices2[f], FeatureImp[indices2[f]]))
        print ("feature name: ", feature_names2[indices2[f]])

#confision matrix for DT1
print("Prediction\n")
bn_matrix1= confusion_matrix(TestLabels, DT_pred1)
print("\nThe confusion matrix is:")
print(bn_matrix1)

FeatureImp=MyDT1.feature_importances_   
indices1= np.argsort(FeatureImp)[::-1]
## print out the important features.....
for f in range(TrainDF.shape[1]):
    if FeatureImp[indices1[f]] > 0:
        print("%d. feature %d (%f)" % (f + 1, indices1[f], FeatureImp[indices1[f]]))
        print ("feature name: ", feature_names1[indices1[f]])

##############################################
##                 Random Forest
##
#################################################################

RF1 = RandomForestClassifier()
RF1.fit(TrainDF, TrainLabels)
RF1_pred=RF1.predict(TestDF)
bn_matrix_RF = confusion_matrix(TestLabels, RF1_pred)
print("\nThe confusion matrix is:")
print(bn_matrix_RF)

################# VIS RF---------------------------------
Features=TrainDF.columns
#Targets=StudentTestLabels_Num

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(RF1.estimators_[0],
               feature_names = Features, 
               #class_names=Targets,
               filled = True)

fig.savefig('RF_Tree')  ## creates png

#####------------------> View estimator Trees in RF

fig2, axes2 = plt.subplots(nrows = 1,ncols = 3,figsize = (10,2), dpi=900)
for index in range(0, 3):
    tree.plot_tree(RF1.estimators_[index],
                   feature_names = Features, 
                   filled = True,
                   ax = axes2[index])

    axes2[index].set_title('Estimator: ' + str(index), fontsize = 11)
fig2.savefig('THREEtrees_RF.png')





