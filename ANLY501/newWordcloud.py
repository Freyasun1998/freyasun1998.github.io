# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 20:32:22 2021

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
from wordcloud import WordCloud, STOPWORDS
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
import seaborn as sns
import sys
from sklearn import metrics
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hc
path="E:/GU/501/data/Textdataset"
print("calling os...")
FileNameList=os.listdir(path)

ListOfCompleteFilePaths=[]
ListOfJustFileNames=[]

for name in os.listdir(path):
    name=name.lower()
    print(path+ "/" + name)
    next=path+ "/" + name
    
    nextname=name  
    print(nextname)  ## ALWAYS check yourself
    
    ListOfCompleteFilePaths.append(next)
    ListOfJustFileNames.append(nextname)

#print("DONE...")
print("full list...")
print(ListOfCompleteFilePaths)
print(ListOfJustFileNames)

#draw wordcloud based on known labels

text1=[]
text2=[]
text3=[]
text4=[]
text5=[]
for i in range(1,10):
        next1=ListOfCompleteFilePaths[i]
        with open(next1,encoding="utf-8") as f:
            text1.append(f.readlines())
for i in range(11,20):
        next2=ListOfCompleteFilePaths[i]
        with open(next2,encoding="utf-8") as f:
            text2.append(f.readlines())
for i in range(21,30):
        next3=ListOfCompleteFilePaths[i]
        with open(next3,encoding="utf-8") as f:
            text3.append(f.readlines())
for i in range(31,40):
        next4=ListOfCompleteFilePaths[i]
        with open(next4,encoding="utf-8") as f:
            text4.append(f.readlines())
for i in range(41,50):
        next4=ListOfCompleteFilePaths[i]
        with open(next4,encoding="utf-8") as f:
            text4.append(f.readlines())
            
wordcloud = WordCloud().generate(str(text1))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud1.png")
wordcloud = WordCloud().generate(str(text2))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud2.png")
wordcloud = WordCloud().generate(str(text3))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud3.png")
wordcloud = WordCloud().generate(str(text4))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud5.png")
wordcloud = WordCloud().generate(str(text4))
plt.figure()
plt.imshow(wordcloud)
plt.axis("off")
plt.savefig("wordcloud5.png")