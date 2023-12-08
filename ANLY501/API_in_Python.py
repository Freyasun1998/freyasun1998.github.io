# -*- coding: utf-8 -*-
"""
Created on Wed Sep 15 00:31:54 2021

@author: Jieyi Sun
"""

import tweepy
from tweepy import OAuthHandler
import json
from tweepy import Stream
from nltk.tokenize import TweetTokenizer
from tweepy.streaming import StreamListener
from nltk.tokenize import word_tokenize
import re
from os import path
import wordcloud
from wordcloud import WordCloud, STOPWORDS

#API Key:
consumer_key="2HlMuB8CWK0eXakreQ2PypDCr"
#API Secret Key:
consumer_secret="snk4MJQTg30rQsOm2KMEfk7Yiox4eVe7tkEIgNfyfcW74IGW1S"
#Access Token:
access_token ="1432687296143319053-Kb2YUvYbg7RHEN21MoMCtLhFsVBbzw"
#Access Token Secret:
access_secret ="zWzWNu6kb42uA5dKLuesioKnhmOQoj66TlpBZYkZPsZni"

auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)
api = tweepy.API(auth)
def Gather(tweet):
    print(json.dumps(tweet))
    
for friend in tweepy.Cursor(api.friends).items():
    Gather(friend._json)

class Listener(StreamListener):
    print("In Listener...") 
    tweet_number=0
def init__(self, max_tweets, hfilename, rawfile):
        self.max_tweets=max_tweets
        print(self.max_tweets
        
def on_data(self, data):
        self.tweet_number+=1 
        print("In on_data", self.tweet_number)
        try:
            print("In on_data in try")
            with open(hfilename, 'a') as f:
                with open(rawfile, 'a') as g:
                    tweet=json.loads(data)
                    tweet_text=tweet["text"]
                    print(tweet_text,"\n")
                    f.write(tweet_text) # the text from the tweet
                    json.dump(tweet, g)  #write the raw tweet
        except BaseException:
            print("NOPE")
            pass
        if self.tweet_number>=self.max_tweets:
            #sys.exit('Limit of '+str(self.max_tweets)+' tweets reached.')
            print("Got ", str(self.max_tweets), "tweets.")
            return False

#method for on_error()
def on_error(self, status):
        print("ERROR")#machi
        print(status)   #401 your keys are not working
        if(status==420):
            print("Error ", status, "rate limited")
            return False

hashname=input("diabetes") 
numtweets=eval('100')
if(hashname[0]=="#"):
    nohashname=hashname[1:] #remove the hash
else:
    nohashname=hashname
    hashname="#"+hashname

filename="file_"+nohashname+".txt"
rawfile="file_rawtweets_"+nohashname+".txt"
twitter_stream = Stream(auth, Listener)
twitter_stream.filter(track=[hashname])
print("Twitter files created....")
linecount=0
hashcount=0
wordcount=0
BagOfWords=[]
BagOfHashes=[]
BagOfLinks=[]

tweetsfile=filename
with open(tweetsfile, 'r') as file:
    for line in file:
        print(line,"\n")
        tweetSplitter = TweetTokenizer(strip_handles=True, reduce_len=True)
        WordList=tweetSplitter.tokenize(line)
        WordList2=word_tokenize(line)
        linecount=linecount+1
        print(WordList)
        print(len(WordList))
        print(WordList[0])
        print(WordList2)
        print(len(WordList2))
        print(WordList2[3:6])
        print("NEXT..........\n")
        regex1=re.compile('^#.+')
        regex2=re.compile('[^\W\d]') #no numbers
        regex3=re.compile('^http*')
        regex4=re.compile('.+\..+')
    for item in WordList:
        if(len(item)>2):
            if((re.match(regex1,item))):
              print(item)
              newitem=item[1:] #remove the hash
              BagOfHashes.append(newitem)
              hashcount=hashcount+1
              elif(re.match(regex2,item)):
                    if(re.match(regex3,item) or re.match(regex4,item)):
                        BagOfLinks.append(item)
                    else:
                        BagOfWords.append(item)
                        wordcount=wordcount+1
                else:
                    pass
            else:
                pass
            
    
print(linecount)            
print(BagOfWords)
print(BagOfHashes)
print(BagOfLinks)
BigBag=BagOfWords+BagOfHashes
#list of words I have seen
seenit=[]
#dict of word counts
WordDict={}
Rawfilename="TwitterResultsRaw.txt"

FILE=open(Freqfilename,"w")
FILE2=open(Rawfilename, "w")
R_FILE=open(Rawfilename,"w")
F_FILE=open(Freqfilename, "w")
IgnoreThese=["and", "And", "AND","THIS", "This", "this", "for", "FOR", "For", 
             "THE", "The", "the", "is", "IS", "Is", "or", "OR", "Or", "will", 
             "Will", "WILL", "God", "god", "GOD", "Bible", "bible", "BIBLE",
             "CanChew", "Download", "free", "FREE", "Free", "will", "WILL", 
             "Will", "hits", "hit", "within", "steam", "Via", "via", "know", "Study",
             "study", "unit", "Unit", "always", "take", "Take", "left", "Left",
             "lot","robot", "Robot", "Lot", "last", "Last", "Wonder", "still", "Still",
             "ferocious", "Need", "need", "food", "Food", "Flint", "MachineCredit",
             "Webchat", "luxury", "full", "fifdh17", "New", "new", "Caroline",
             "Tirana", "Shuayb", "repro", "attempted", "key", "Harrient", 
             "Chavez", "Women", "women", "Mumsnet", "Ali", "Tubman", "girl","Girl",
             "CSW61", "IWD2017", "Harriet", "Great", "great", "single", "Single", 
             "tailoring", "ask", "Ask"]

###Look at the words
for w in BigBag:
    if(w not in IgnoreThese):
        rawWord=w+" "
        R_FILE.write(rawWord)
        if(w in seenit):
            print(w, seenit)
            WordDict[w]=WordDict[w]+1 #increment the times word is seen
        else:
            ##add word to dict and seenit
            seenit.append(w)
            WordDict[w]=1

print(WordDict) 
print(seenit)
print(BagOfWords)

for key in WordDict:
    print(WordDict[key])
    if(WordDict[key]>1):
        if(key not in IgnoreThese):
            print(key)
            Key_Value=key + "," + str(WordDict[key]) + "\n"
            F_FILE.write(Key_Value)
FILE.close()
FILE2.close()
R_FILE.close()
F_FILE.close()
d = path.dirname(__file__)
Rawfilename="TwitterResultsRaw.txt"
# Read the whole text.
text = open(path.join(d, Rawfilename)).read()
print(text)
lines) 
##---------
wordcloud = WordCloud().generate(text)
# Open a plot of the generated image.
figure(figsize = (20,2))
plt.figure(figsize=(50,40))
plt.imshow(wordcloud)
           #, aspect="auto")
plt.axis("off")













