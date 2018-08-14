from twitter import Twitter, OAuth, TwitterHTTPError, TwitterStream
from langdetect import detect
from datetime import datetime
import time
import tweepy
import pyquery
import datetime
import time
import lxml
import sys

try:
    import json
except ImportError:
    import simplejson as json

import sys,getopt,datetime,codecs
if sys.version_info[0] < 3:
    import got
else:
    import got3 as got

tweetsCriteria = got.manager.TweetCriteria()

tweetsCriteria = tweetsCriteria.setMaxTweets(2000)

tweetsCriteria = tweetsCriteria.setQuerySearch("Vodafone")

tweetsCriteria = tweetsCriteria.setSince(sys.argv[1])

tweetsCriteria = tweetsCriteria.setUntil(sys.argv[2])

tweets = got.manager.TweetManager.getTweets(tweetsCriteria)

files = open("vdf_tweets1.txt","a")

tweet_dict = {}
for tweet in tweets:
    year = tweet.date.year
    month = tweet.date.month
    day = tweet.date.day
    
    date = str(year)+"-"+str(month)+"-"+str(day)

    date = datetime.datetime.strptime(date,'%Y-%m-%d')
    date = time.mktime(date.timetuple())

    ID = tweet.id
    txt = tweet.text
    tweet_dict[ID]=txt
    try:
    	lang = detect(txt)
    except:
    	lang = "Nan"

    full_str = str(ID)+" "+str(txt)+" "+str(year)+"-"+str(month)+"-"+str(day)+" "+str(lang)+"\n"
    files.write(full_str)

files.close()

