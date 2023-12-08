# -*- coding: utf-8 -*-
"""
Created on Thu Sep 30 03:39:12 2021

@author: Jieyi Sun
"""

import pandas as pd
import os
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
path="E:/GU/501/data/"
filelocation="News.csv"
CSV_DF=pd.read_csv(path+filelocation)
print(CSV_DF)
My_FILE=open(path+filelocation, "r")
for next_row in My_FILE:
    print(next_row)
My_FILE.close()
My_Content_List=[]
My_Labels_List=[]
with open(path+filelocation, "r") as My_FILE:
    next(My_FILE) 
    for next_row in My_FILE:
        print(next_row)
        Row_Elements=next_row.split(",")
        print(Row_Elements)
        print("The label is: \n", Row_Elements[0])
        print("\nThe review content is:\n",Row_Elements[1])
        ## OK! Now that we know this works, we can BUILD
        ## our lists....
        My_Content_List.append(Row_Elements[1])
        My_Labels_List.append(Row_Elements[0])
## Let's see what we built....
print(My_Content_List)
print(My_Labels_List)
MyCV_content=CountVectorizer(input='content',
                        stop_words='english',
                        #max_features=100
                        )
My_DTM2=MyCV_content.fit_transform(My_Content_List)
ColNames=MyCV_content.get_feature_names()
print("The vocab is: ", ColNames, "\n\n")
My_DF_content=pd.DataFrame(My_DTM2.toarray(),columns=ColNames)
print(My_DF_content)
print(My_Labels_List)
My_DF_content.insert(loc=0, column='LABEL', value=My_Labels_List)
print(My_DF_content)
My_DF_content.to_csv(path+'MyClean_CSV_Data.csv', index=False)

