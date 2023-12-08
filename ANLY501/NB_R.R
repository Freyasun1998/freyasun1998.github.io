# Naive Bayes
# Record Data
library(NLP)
library(tm)
#install.packages("tm")
library(stringr)
library(RColorBrewer)
library(wordcloud)
# ONCE: install.packages("Snowball")
## NOTE Snowball is not yet available for R v 3.5.x
## So I cannot use it  - yet...
##library("Snowball")
##set working directory
## ONCE: install.packages("slam")
library(Matrix)
library(slam)
library(quanteda)
## ONCE: install.packages("quanteda")
## Note - this includes SnowballC
library(SnowballC)
library(arules)
##ONCE: install.packages('proxy')
library(proxy)
library(cluster)
library(stringi)
library(proxy)
library(Matrix)
library(tidytext) # convert DTM to DF
library(plyr) ## for adply
library(ggplot2)
library(factoextra) # for fviz
library(mclust) # for Mclust EM clustering

library(naivebayes)
#Loading required packages
#install.packages('tidyverse')
library(tidyverse)
#install.packages('ggplot2')
library(ggplot2)
#install.packages('caret')
library(caret)
#install.packages('caretEnsemble')
library(caretEnsemble)
#install.packages('psych')
library(psych)
#install.packages('Amelia')
library(Amelia)
#install.packages('mice')
library(mice)
#install.packages('GGally')
library(GGally)
library(e1071)
setwd("E:/GU/501/NB_SVM")
## Read in the dataset
DataFile="Cleaned_Diabetes_Dataset.csv"
head(DF<-read.csv(DataFile))
## MAKE test and train data
str(DF)
(Size <- (as.integer(nrow(DF)/4)))  ## Test will be 1/4 of the data
(SAMPLE <- sample(nrow(DF), Size, replace = FALSE))

(DF_Test<-DF[SAMPLE, ])
(DF_Train<-DF[-SAMPLE, ])
## REMOVE the labels and KEEP THEM
########### REMOVE AND SAVE LABELS...
## Copy the Labels
(DF_Test_Labels <- DF_Test$Outcome)
## Remove the labels
DF_Test_NL<-DF_Test[ , -which(names(DF_Test) %in% c("Outcome"))]
## Check size
(ncol(DF_Test_NL))
## Train...--------------------------------
## Copy the Labels
(DF_Train_Labels <- DF_Train$Outcome)
## Remove the labels
DF_Train_NL<-DF_Train[ , -which(names(DF_Train) %in% c("Outcome"))]
## Check size
(ncol(DF_Train_NL))

#Tune model-laplace (cited by Zhaoyuan Qiu)
Accuracy=c()
Value_range<-seq(0,1000,10)
for(laplace in Value_range){
  model<-naiveBayes(Outcome~.,data=DF_Train,laplace=laplace)
  prediction<-predict(model,DF_Test_NL)
  Accuracy<-c(Accuracy,sum(prediction==DF_Test$Outcome)/nrow(DF_Test))
}
plot(Value_range,Accuracy,'l',xlab='Laplace Value')
#Tune model-eps
Accuracy=c()
Value_range<-seq(0,1,0.05)
for(eps in Value_range){
  model<-naiveBayes(Outcome~.,data=DF_Train,eps=eps)
  prediction<-predict(model,DF_Test_NL)
  Accuracy<-c(Accuracy,sum(prediction==DF_Test$Outcome)/nrow(DF_Test))
}
plot(Value_range,Accuracy,'l',xlab='Eps Value')
#Tune model-threshold
Accuracy=c()
Value_range<-seq(0,200,20)
for(threshold in Value_range){
  model<-naiveBayes(Outcome~.,data=DF_Train,threshold=threshold)
  prediction<-predict(model,DF_Test_NL)
  Accuracy<-c(Accuracy,sum(prediction==DF_Test$Outcome)/nrow(DF_Test))
}
plot(Value_range,Accuracy,'l',xlab='threshold value')
#######################
####### RUN Naive Bayes ---
(NB<-naiveBayes(DF_Train_NL, 
                DF_Train_Labels, 
                laplace = 1))

NB_Pred <- predict(NB, DF_Test_NL)
#NB
table(NB_Pred,DF_Test_Labels)
#Confusion Matrix
library(pheatmap)
confusion_mx<-table(DF_Test$Outcome,NB_Pred)
pheatmap(confusion_mx,cluster_cols=F,cluster_rows=F,display_numbers=T,number_format = "%.f",legend=F,main='Confusion Matrix')
DF_Test['prediction']=NB_Pred
sum(DF_Test$prediction==DF_Test$Outcome)/nrow(DF_Test)
#The accuracy is 0.766 with the default params.
barplot(NB$apriori/sum(NB$apriori),main = 'Apriori Probability')
