import pandas as pd
import tweepy
import constants as ct
from textblob import TextBlob

flag = False
df = pd.read_csv('companylist.csv', usecols=[0])

while flag is False:
    symbol = raw_input('Enter a stock symbol to retrieve data from: ').upper()
    for index in range(len(df)):
        if df['Symbol'][index] == symbol:
            flag = True


auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
auth.set_access_token(ct.access_token, ct.access_token_secret)
user = tweepy.API(auth)

tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en').items(ct.num_of_tweets)

analysis_tw = list()
for tweet in tweets:
    analysis_tw.append(tweet.full_text.strip())


for tw in analysis_tw:
    blob = TextBlob(tw)
    for sentence in blob.sentences:
        print sentence.sentiment.polarity