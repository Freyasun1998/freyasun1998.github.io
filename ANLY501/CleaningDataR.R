library(ggplot2)
library(plotly)
setwd("E:/GU/501/data")
df<-read.csv("diabetes-dataset.csv")
#Remove the columns we don not want
df<-df[,-1]
#Make tables of all the columns
lapply(df, table)
lapply(df,summary)
#Replace all the 0 value in the variables with NA
Outcome<-df$Outcome
df_variable<-df[,-8]
df_variable[df_variable==0]<-NA
df1<-data.frame(df_variable,Outcome)
#Replace NA value of the variables with Median values
lapply(df1, summary)
df1$Glucose[is.na(df1$Glucose)]<-117
df1$BloodPressure[is.na(df1$BloodPressure)]<-72.0
df1$SkinThickness[is.na(df1$SkinThickness)]<-29.00
df1$Insulin[is.na(df1$Insulin)]<-126.00
df1$BMI[is.na(df1$BMI)]<-32.40
df1$Age<-as.numeric(df1$Age)
write.csv(df1,"Cleaned_Diabetes_Dataset.csv")
