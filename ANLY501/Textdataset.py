# -*- coding: utf-8 -*-
"""
Created on Mon Oct 11 00:13:36 2021

@author: Jieyi Sun
"""

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem import PorterStemmer
import numpy as np
import pandas as pd
import os
import re
import warnings
warnings.filterwarnings('ignore')
#Get the text data
path="E:/GU/501/data/Textdataset/"
## Get the text data first
f=open("DiabetesCauses1.txt",'r',encoding='utf-8',errors='ignore')

MyFileNameList=[]
FileNames=[]
for nextfile in os.listdir(path):
    print(nextfile)
for nextfile in os.listdir(path):
    fullpath=path+"/"+nextfile
    print(fullpath)
for nextfile in os.listdir(path):
    fullpath=path+"/"+nextfile
    #print(fullpath)
    MyFileNameList.append(fullpath)
    ## let's place the files names (not the path too)
    ## into our other list
    FileNames.append(nextfile)
print(MyFileNameList)
print("\\\\\\\\\\\\\\\/n")
print(FileNames)


## Using CountVectorizer to create a Document Term Matrix
MyCV=CountVectorizer(input='filename',
                        stop_words='english',
                        max_features=400,
                        encoding="ISO-8859-1"
                        )
My_DTM=MyCV.fit_transform(MyFileNameList)
MyColumnNames=MyCV.get_feature_names()
print("The vocab is: ", MyColumnNames, "\n\n")
##use pandas to create data frames
My_DF=pd.DataFrame(My_DTM.toarray(),columns=MyColumnNames)
print(My_DF)
print(FileNames)
#add the labels
for filename in FileNames:
    print(filename) ## Make sure you can loop
for filename in FileNames:
    print(type(filename))
    ## remove the number and the .txt from each file name
    newName=filename.split(".")
    print(newName)
    print(newName[0])
    
CleanNames=[]
for filename in FileNames:
    ## remove the number and the .txt from each file name
    newName=filename.split(".")
    print(newName[0])
    ## remove any numbers
    newName2=re.sub(r"[^A-Za-z\-]", "", newName[0])
    print(newName2)
    CleanNames.append(newName2)
print(CleanNames)
print(My_DF)
My_DF["LABEL"]=CleanNames
print(My_DF)
My_DF.to_csv(path+'Textdataset.csv', index=False)