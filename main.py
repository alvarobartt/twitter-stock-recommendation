import pandas as pd
import tweepy
import constants as ct
from textblob import TextBlob
from Tweet import Tweet
import fix_yahoo_finance as yf
import datetime as dt

actual_date = dt.date.today()
last_month_date = actual_date - dt.timedelta(days=30)
flag = False
df = pd.read_csv('companylist.csv', usecols=[0])

while flag is False:
    symbol = raw_input('Enter a stock symbol to retrieve data from: ').upper()
    for index in range(len(df)):
        if df['Symbol'][index] == symbol:
            flag = True

data = yf.download("SPY", start="2017-01-01", end="2017-04-30")
df = pd.DataFrame(data=data)

auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
auth.set_access_token(ct.access_token, ct.access_token_secret)
user = tweepy.API(auth)

tweets = tweepy.Cursor(user.search, q=symbol, tweet_mode='extended', lang='en').items(ct.num_of_tweets)

analysis_tw = list()
for tweet in tweets:
    analysis_tw.append(tweet.full_text.strip())


for tw in analysis_tw:
    blob = TextBlob(tw)
    polarity = 0
    for sentence in blob.sentences:
        polarity += sentence.sentiment.polarity
    t = Tweet(tw, polarity)