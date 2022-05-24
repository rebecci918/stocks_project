#This code scrapes twitter for tweets from the day (must change start date to today), 
#runs sentiment analysis, and counts number of positive and negative tweets.  
import pandas as pd
import numpy as np
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path
import tweepy
import keys
api = tweepy.Client(keys.bearer, wait_on_rate_limit=True)


queries = ['SPX500', 'SP500', 'SPX', 'stocks', 'MSFT', 'AAPL', 'AMZN', 'FB', 'GOOG']

query = np.nan
start_time = "2022-04-26T00:00:00z"
count = 10
mylist = list()
for i in range(len(queries)):
    query = queries[i]
    tweets = api.search_recent_tweets(query=query, max_results = count, start_time = start_time, tweet_fields = ["text"])
    for tweet in tweets.data:
        mylist.append(tweet.text)


tweets1 = pd.DataFrame()
tweets1['Text'] = mylist
tweets1['Sentiment'] = np.nan

sia = SentimentIntensityAnalyzer()

tweets1['scores'] = tweets1['Text'].apply(lambda review: sia.polarity_scores(review))
tweets1['compound']  = tweets1['scores'].apply(lambda score_dict: score_dict['compound'])

for i in range(len(tweets1)):
    if tweets1.loc[i, 'compound'] > 0:
        tweets1.loc[i, 'Sentiment'] = 'positive'
    elif tweets1.loc[i, 'compound'] < 0:
        tweets1.loc[i,'Sentiment'] = 'negative'

todays_stat = pd.DataFrame(np.zeros((1,3)))
todays_stat.columns = ['Positive','Negative','Ratio']
for i in range(len(tweets1)):
    if tweets1.loc[i, 'Sentiment'] == 'positive':
                todays_stat.loc[0, 'Positive'] += 1
    elif tweets1.loc[i, 'Sentiment'] == 'negative':
                todays_stat.loc[0, 'Negative'] += 1

todays_stat.loc[0,'Ratio'] = todays_stat.loc[0,'Positive']/todays_stat.loc[0,'Negative']

filepath = Path('C:/Users/mekak/OneDrive/Documents/Spring_2022/DSC_325/stocks_project/todays_stat.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
todays_stat.to_csv(filepath) 