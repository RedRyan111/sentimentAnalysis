from praw.models import MoreComments
from datetime import date
import datetime
import pandas as pd
import numpy as np
import webbrowser
import urllib.request
import praw
import json
import sys
import re


com_freq = "https://api.pushshift.io/reddit/search/comment/?q=vodafone&aggs=created_utc&after={0}d&before={1}d&frequency=hour&size=50".format(sys.argv[0],sys.argv[1])

sub_freq = "https://api.pushshift.io/reddit/search/submission/?q=vodafone&aggs=created_utc&after={0}d&before={1}d&frequency=hour&size=50".format(sys.argv[0],sys.argv[1])


df_com = pd.DataFrame({'created_utc':0,'text':""},index=[0])

df_sub = pd.DataFrame({'created_utc':0,'text':"",'body':""},index=[0])

cols = ["created_utc",'time']

with urllib.request.urlopen(com_freq) as url:
    data = json.loads(url.read().decode())

    data1 = data['data']

    data2 = data['aggs']['created_utc'] 
    
    cur_ind = 0
    cur_count = 0
    for i in range(len(data1)):
        if( cur_count == data2[cur_ind]['doc_count']):
            cur_count = 0
            cur_ind +=1

        time_list = {}
        time_list['created_utc'] = data2[cur_ind]['key']
        time_list['text'] = data1[i]['body']
        
        cur_count+=1

        df = pd.DataFrame(time_list,index = [0])
        df_com = df_com.append(df,cols)

df_com = df_com.iloc[1:]

df_sub = df_sub.iloc[1:]

df_com.to_csv('reddit_com.csv',header=False,index=False,mode='a')

df_sub.to_csv('reddit_sub.csv',header=False,index=False,mode='a')

