# -*- coding: utf-8 -*-
"""
Created on Mon Sep 20 10:50:36 2021

@author: Jieyi Sun
"""

import tweepy
#API Key:
consumer_key="2HlMuB8CWK0eXakreQ2PypDCr"
#API Secret Key:
consumer_secret="snk4MJQTg30rQsOm2KMEfk7Yiox4eVe7tkEIgNfyfcW74IGW1S"
#Access Token:
access_token ="1432687296143319053-Kb2YUvYbg7RHEN21MoMCtLhFsVBbzw"
#Access Token Secret:
access_secret ="zWzWNu6kb42uA5dKLuesioKnhmOQoj66TlpBZYkZPsZni"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)
tweets_UK=api.user_timeline(id='Diabetes UK',count=30)
for tweet in tweets_UK:
    print(tweets_UK)
    
tweets_US=api.user_timeline(id='Roche Diabetes Care US',count=30)
for tweet in tweets_US:
    print(tweets_US)

q=input("diabetes")
filename="rawtext_"+q+".txt"
f=open(filename,'a',encoding='utf-8')
response1=tweepy.Cursor(api.search,q='diabetes',lang="en").items(300)
matchcount=0
for match in response1:
    f.write(match.text)
    f.write('\n')
    matchcount=matchcount+1
print(matchcount)
print(match.text)
