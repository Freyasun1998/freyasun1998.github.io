##############   Using the News  API   ##################
library("httr")
library("jsonlite")
library("xml2")

#Get the data
##Build the URL
base<-"https://newsapi.org/v2/everything"
q<-"diabetes"
from<-"2021-08-15"
to<-"2021-09-13"
APIKey<-"99821579a2ad4d238525a36999203262"

format<-"text/csv"
call1 <- paste(base,"?",
               "format","=",format,"&",
               "q", "=", q, "&",
               "from","=",from,"&",
               "to","=",to,"&",
               "APIKey", "=",APIKey,
               sep="")
call1

NewsAPI_Call<-httr::GET(call1)
NewsAPI_Call
MYDF<-httr::content(NewsAPI_Call)
MYDF
MYDF<-MYDF$articles
df = as.data.frame(matrix(nrow=0,ncol=7))
colnames(df)<-c("author","title","description","url","urlTolmage","PublishedAt","content")
for (i in 1:20){
  MYDF[[i]]["source"] <- NULL
}

MYDF1<-MYDF[[1]]
MYDF2<-MYDF[[2]]
MYDF3<-MYDF[[3]]
MYDF4<-MYDF[[4]]
MYDF5<-MYDF[[5]]
MYDF6<-MYDF[[6]]
MYDF7<-MYDF[[7]]
MYDF8<-MYDF[[8]]
MYDF9<-MYDF[[9]]
MYDF10<-MYDF[[10]]
MYDF11<-MYDF[[11]]
MYDF12<-MYDF[[12]]
MYDF13<-MYDF[[13]]
MYDF14<-MYDF[[14]]
MYDF15<-MYDF[[15]]
MYDF16<-MYDF[[16]]
MYDF17<-MYDF[[17]]
MYDF18<-MYDF[[18]]
MYDF19<-MYDF[[19]]
MYDF20<-MYDF[[20]]
MYDFnew<-rbind(MYDF1,MYDF2,MYDF3,MYDF4,MYDF5,MYDF6,MYDF7,MYDF8,MYDF9,MYDF10,MYDF11,MYDF12,MYDF13,MYDF14,MYDF15,MYDF16,MYDF17,MYDF18,MYDF19,MYDF20)
##Print to a file
NewsName="News.csv"
##Start the file
NewsFile<-file(NewsName)
write.csv(MYDFnew,NewsFile,row.names = FALSE)
