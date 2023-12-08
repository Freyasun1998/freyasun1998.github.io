## do library
#install.packages("twitteR")
#install.packages("ROAuth")
#install.packages("rtweet")
library(rtweet)
library(twitteR)
library(ROAuth)
library(arules)
library(jsonlite)
library(selectr)
library(rvest)
library(xml2)
#install.packages("streamR")
library(streamR)
#install.packages("rjson")
library(rjson)
#install.packages("tokenizers")
library(tokenizers)
library(tidyverse)
library(plyr)
library(dplyr)
library(ggplot2)
#install.packages("syuzhet")  ## sentiment analysis
library(syuzhet)
library(stringr)
library(arulesViz)
library(igraph)
library(httpuv)
library(openssl)
##############  Using twittR ###############################
library(devtools)
devtools::install_version("httr", version="0.6.0", repos="http://cran.us.r-project.org")
devtools::install_version("httr", version="0.6.0", repos="http://cran.us.r-project.org")
devtools::install_version("twitteR", version="1.1.8", repos="http://cran.us.r-project.org")
consumerKey="2HlMuB8CWK0eXakreQ2PypDCr"
consumerSecret="snk4MJQTg30rQsOm2KMEfk7Yiox4eVe7tkEIgNfyfcW74IGW1S"
access_Token="1432687296143319053-Kb2YUvYbg7RHEN21MoMCtLhFsVBbzw"
access_Secret="zWzWNu6kb42uA5dKLuesioKnhmOQoj66TlpBZYkZPsZni"

requestURL='https://api.twitter.com/oauth/request_token'
accessURL='https://api.twitter.com/oauth/access_token'
authURL='https://api.twitter.com/oauth/authorize'

my_oauth <- OAuthFactory(consumerKey = consumerKey,
                                                           consumerSecret = consumerSecret,
                                                           requestURL = requestURL,
                                                           accessURL = accessURL,
                                                           authURL = authURL)
library(base64enc)
setup_twitter_oauth(consumerKey,consumerSecret,access_Token,access_Secret)
Search<-twitteR::searchTwitter("#diabetes", n=100, since = "2021-01-01",lang="en")
Search2 <- searchTwitter("diabetes", 
                                                    n = 100, 
                                                    lang = "en",
                                                    since='2021-01-01', ## need special account
                                                    until='2021-05-02'  ## need special account
                                                    ) 
                          
Search_DF <- twitteR::twListToDF(Search)
TransactionTweetsFile ="TweetResults.csv"

Search_DF$text[100]
#make data frame
Search2_DF <- do.call("rbind", lapply(Search2, as.data.frame))
tokenize_tweets(x, lowercase = TRUE, stopwords = NULL, strip_punct = TRUE, 
                                 strip_url = FALSE, simplify = FALSE)
                

tokenize_tweets(Search2_DF$text[1],stopwords = stopwords::stopwords("en"), 
                               lowercase = TRUE,  strip_punct = TRUE, 
                               strip_url = TRUE, simplify = TRUE)
## Start the file
setwd("E:/GU/501/ARM")
Trans <- file(TransactionTweetsFile)
## Tokenize to words 
Tokens<-tokenizers::tokenize_words(
  Search_DF$text[1],stopwords = stopwords::stopwords("en"), 
  lowercase = T,  strip_punct = T, strip_numeric = T,
  simplify = T)
## Write tokens
cat(unlist(Tokens), "\n", file=Trans, sep=",")
close(Trans)
## Append remaining lists of tokens into file
## Recall - a list of tokens is the set of words from a Tweet
Trans <- file(TransactionTweetsFile, open = "a")
for(i in 2:nrow(Search_DF)){
  Tokens<-tokenize_words(Search_DF$text[i],
                         stopwords = stopwords::stopwords("en"), 
                         lowercase = T,  
                         strip_punct = T, 
                         simplify = T)
  
  cat(unlist(Tokens), "\n", file=Trans, sep=",")
  cat(unlist(Tokens))
}
close(Trans)
######### Read in the tweet transactions
TweetTrans <- read.transactions("TweetResults.csv",
                                 rm.duplicates = FALSE, 
                                 format = "basket",
                                 sep=","
                                 ## cols = 
)
inspect(TweetTrans)
## See the words that occur the most
Sample_Trans <- sample(TweetTrans, 3)
summary(Sample_Trans)
## Read the transactions data into a dataframe
TweetDF <- read.csv("TweetResults.csv", 
                    header = FALSE, sep = ",")
head(TweetDF)
str(TweetDF)
## Convert all columns to char 
TweetDF<-TweetDF %>%
  mutate_all(as.character)
str(TweetDF)
# We can now remove certain words
TweetDF[TweetDF == "t.co"] <- ""
TweetDF[TweetDF == "rt"] <- ""
TweetDF[TweetDF == "http"] <- ""
TweetDF[TweetDF == "https"] <- ""
TweetDF
## Clean with grepl - every row in each column
MyDF<-NULL
MyDF2<-NULL
for (i in 1:ncol(TweetDF)){
  MyList=c() 
  MyList2=c() # each list is a column of logicals ...
  MyList=c(MyList,grepl("[[:digit:]]", TweetDF[[i]]))
  MyDF<-cbind(MyDF,MyList)  ## create a logical DF
  MyList2=c(MyList2,(nchar(TweetDF[[i]])<4 | nchar(TweetDF[[i]])>11))
  MyDF2<-cbind(MyDF2,MyList2) 
  ## TRUE is when a cell has a word that contains digits
}
## For all TRUE, replace with blank
TweetDF[MyDF] <- ""
TweetDF[MyDF2] <- ""
head(TweetDF,10)
# Now we save the dataframe using the write table command 
write.table(TweetDF, file = "UpdatedTweetFile.csv", col.names = FALSE, 
            row.names = FALSE, sep = ",")
TweetTrans <- read.transactions("UpdatedTweetFile.csv", sep =",", 
                                format("basket"),  rm.duplicates = TRUE)
inspect(TweetTrans)
