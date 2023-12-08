# -*- coding: utf-8 -*-
"""
Created on Sat Oct  2 13:06:06 2021

@author: Jieyi Sun
"""

import nltk
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
## For Stemming
from nltk.stem import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize
import os
import re   ## for regular expressions
from mpl_toolkits.mplot3d import Axes3D
#from nltk.stem.porter import PorterStemmer
path="E:/GU/501/discussion3/question1/text/"
print("calling os...")
FileNameList=os.listdir(path)
print(type(FileNameList))
print(FileNameList)
ListOfCompleteFilePaths=[]
ListOfJustFileNames=[]
for name in os.listdir(path):
    ## BUILD the names dynamically....
    name=name.lower()
    print(path+ "/" + name)
    next=path+ "/" + name
    
    nextnameL=[re.findall(r'[a-z]+', name)[0]]  
    nextname=nextnameL[0]   ## Keep just the name
    print(nextname)  ## ALWAYS check yourself
    
    ListOfCompleteFilePaths.append(next)
    ListOfJustFileNames.append(nextname)
print("full list...")
print(ListOfCompleteFilePaths)
print(ListOfJustFileNames)
A_STEMMER=PorterStemmer()
print(A_STEMMER.stem("fishers"))
def MY_STEMMER(str_input):
    ## Only use letters, no punct, no nums, make lowercase...
    words = re.sub(r"[^A-Za-z\-]", " ", str_input).lower().split()
    words = [A_STEMMER.stem(word) for word in words] ## Use the Stemmer...
    return words
MyVectCount=CountVectorizer(input='filename',   ## 'content'
                        stop_words='english',  ## and, of, the, is, ...
                        max_features=100
                        )

## Tf-idf vectorizer
MyVectTFIdf=TfidfVectorizer(input='filename',
                        stop_words='english',
                        max_features=100
                        )
## Create a CountVectorizer object that you can use with the Stemmer
MyCV_Stem = CountVectorizer(input="filename", 
                        stop_words='english', 
                        tokenizer=MY_STEMMER, ## hikes and hike --> one col hik
                        lowercase=True)
DTM_Count=MyVectCount.fit_transform(ListOfCompleteFilePaths)
DTM_TF=MyVectTFIdf.fit_transform(ListOfCompleteFilePaths)
DTM_stem=MyCV_Stem.fit_transform(ListOfCompleteFilePaths)
ColumnNames=MyVectCount.get_feature_names()
print("The vocab is: ", ColumnNames, "\n\n")
ColNamesStem=MyCV_Stem.get_feature_names()
print("The stemmed vocab is\n", ColNamesStem)
DF_Count=pd.DataFrame(DTM_Count.toarray(),columns=ColumnNames)
DF_TF=pd.DataFrame(DTM_TF.toarray(),columns=ColumnNames)
DF_stem=pd.DataFrame(DTM_stem.toarray(),columns=ColNamesStem)
print(DF_Count)
print(DF_TF)
print(DF_stem)
MyDict={}
for i in range(0, len(ListOfJustFileNames)):
    MyDict[i] = ListOfJustFileNames[i]
print("MY DICT:", MyDict)
        
DF_Count=DF_Count.rename(MyDict, axis="index")
print(DF_Count)
DF_TF=DF_TF.rename(MyDict, axis="index")
print(DF_TF)
from sklearn.cluster import KMeans
import numpy as np
kmeans_object_Count = sklearn.cluster.KMeans(n_clusters=2)
print(kmeans_object_Count)
kmeans_object_Count.fit(DF_Count)
# Get cluster assignment labels
labels = kmeans_object_Count.labels_
prediction_kmeans = kmeans_object_Count.predict(DF_Count)
print(labels)
print(prediction_kmeans)
Myresults = pd.DataFrame([DF_Count.index,labels]).T
print(Myresults)

import os
Texts=[]
folderpath=r"E:/GU/501/discussion3/question1/text"
for file in os.listdir(folderpath):
    print(file)
    filepath=os.path.join(folderpath,file)
    with open(filepath,"r",encoding='utf-8') as f:
        Texts.append(f.readlines())
Flowers1=Texts[1:3]
Flowers2=Texts[4:6]
import wordcloud
import numpy as np
from PIL import Image
mask=np.array(Image.open(r"E:/GU/501/discussion3/question1/Flower1.jpg"))
Flowers1=[Flower1[0] for Flower1 in Flowers1]
Flowers1=" ".join(Flowers1)
w=wordcloud.WordCloud(font_path=r"msyhl.ttc",mask=mask,width=600,height=600,background_color="white")

w.generate(Flowers1)
w.to_file(r"flower1.png")

mask=np.array(Image.open(r"E:/GU/501/discussion3/question1/Flower2.jpg"))
Flowers2=[Flower2[0] for Flower2 in Flowers2]
Flowers2=" ".join(Flowers2)
w=wordcloud.WordCloud(font_path=r"msyhl.ttc",mask=mask,width=600,height=600,background_color="white")

w.generate(Flowers2)
w.to_file(r"flower2.png")


