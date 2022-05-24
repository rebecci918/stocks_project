
# This code runs sentiment analysis of each tweet, converts data to categorical, and combines dataframes 
# into new dataframe to count positive and negative tweets, and S&P change based on matching date. 

import pandas as pd
import numpy as np
from datetime import datetime
import nltk
nltk.download('vader_lexicon')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from pathlib import Path

labelled = pd.read_csv ('tweets_labelled2.csv', sep = ';', on_bad_lines= 'skip')

sia = SentimentIntensityAnalyzer()

labelled['scores'] = labelled['text'].apply(lambda review: sia.polarity_scores(review))
labelled['compound']  = labelled['scores'].apply(lambda score_dict: score_dict['compound'])

for i in range(len(labelled)):
    if labelled.loc[i, 'compound'] > 0:
        labelled.loc[i, 'sentiment'] = 'positive'
    elif labelled.loc[i, 'compound'] < 0:
        labelled.loc[i,'sentiment'] = 'negative'

#importing S&P 500 data
SnP = pd.read_csv('S&P_change.csv')

#changing change to categorical 
SnP['ChangeC'] = np.nan
SnP['Change'] = SnP['Change'].astype(float)

for i in range(len(SnP)): 
    if SnP.loc[i,'Change'] >= 0 and SnP.loc[i,'Change'] < 28.75:
        SnP.loc[i, 'ChangeC'] = 'small_increase'
    elif SnP.loc[i,'Change'] >= 28.75:
        SnP.loc[i, 'ChangeC'] = 'large_increase'
    elif SnP.loc[i,'Change'] < 0 and SnP.loc[i,'Change'] >= -28.75:
        SnP.loc[i, 'ChangeC'] = 'small_decrease'
    else:
        SnP.loc[i, 'ChangeC'] = 'large_decrease'

#adding SnP change to tweets data.
labelled["Change"] = np.nan

for i in range(len(labelled)):
    labelled.loc[i,'created_at'] = labelled.loc[i,'created_at'][:10]

for i in range(len(labelled)):
    labelled.loc[i,'created_at'] = datetime.fromisoformat(labelled.loc[i,'created_at']).timestamp()

for i in range(len(SnP)):
    SnP.loc[i,'Date'] = datetime.fromisoformat(SnP.loc[i,'Date']).timestamp()

for i in range(len(SnP)):
    for j in range(len(labelled)):
        if labelled.loc[j,'created_at'] == SnP.loc[i,'Date']:
            labelled.loc[j,'Change'] = SnP.loc[i,'ChangeC']

a = labelled['created_at'].unique()
dates = sorted(a)


df = pd.DataFrame(np.zeros((77,4)))
df.columns = ['Date', 'Positive', 'Negative', 'Change']

for i in range(len(dates)):
    df.loc[i,'Date'] = dates[i]

for i in range(len(df)):
    for j in range(len(labelled)):
        if df.loc[i,'Date'] == labelled.loc[j,'created_at']:
            if labelled.loc[j, 'sentiment'] == 'positive':
                df.loc[i, 'Positive'] += 1
            elif labelled.loc[j, 'sentiment'] == 'negative':
                df.loc[i, 'Negative'] += 1
            df.loc[i, 'Change'] = labelled.loc[j, 'Change']

df = df.dropna()
df = df.reset_index()

filepath = Path('C:/Users/mekak/OneDrive/Documents/Spring_2022/DSC_325/stocks_project/mydata.csv')  
filepath.parent.mkdir(parents=True, exist_ok=True)  
df.to_csv(filepath) 