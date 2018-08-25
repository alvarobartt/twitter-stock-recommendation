import datetime as dt
import math

import fix_yahoo_finance as yf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tweepy
from matplotlib import style
from sklearn import preprocessing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob

import constants as ct
from Tweet import Tweet

style.use('ggplot')

flag = False
df = pd.read_csv('companylist.csv', usecols=[0])

while flag is False:
    symbol = raw_input('Enter a stock symbol to retrieve data from: ').upper()
    for index in range(len(df)):
        if df['Symbol'][index] == symbol:
            flag = True
            symbol = symbol

actual_date = dt.date.today()
past_date = actual_date - dt.timedelta(days=366)

actual_date = actual_date.strftime("%Y-%m-%d")
past_date = past_date.strftime("%Y-%m-%d")

data = yf.download("AAPL", start=past_date, end=actual_date)
df = pd.DataFrame(data=data)

df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
df['HighLoad'] = (df['High'] - df['Close']) / df['Close'] * 100.0
df['Change'] = (df['Close'] - df['Open']) / df['Open'] * 100.0

df = df[['Close', 'HighLoad', 'Change', 'Volume']]

forecast_col = 'Close'
forecast_out = int(math.ceil(0.01*len(df)))
df['Label'] = df[[forecast_col]].shift(-forecast_out)

X = np.array(df.drop(['Label'], axis=1))
X = preprocessing.scale(X)
X_forecast = X[-forecast_out:]
X = X[:-forecast_out]

df.dropna(inplace=True)
y = np.array(df['Label'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

clf = LinearRegression(n_jobs=-1)
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
forecast = clf.predict(X_forecast)

df['Prediction'] = np.nan

last_date = df.iloc[-1].name
last_date = dt.datetime.strptime(str(last_date), "%Y-%m-%d %H:%M:%S")

for pred in forecast:
    last_date += dt.timedelta(days=1)
    df.loc[last_date.strftime("%Y-%m-%d")] = [np.nan for _ in range(len(df.columns) - 1)] + [pred]

df['Close'].plot(color='black')
df['Prediction'].plot(color='green')
plt.legend(loc=4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
auth.set_access_token(ct.access_token, ct.access_token_secret)
user = tweepy.API(auth)

tweets = tweepy.Cursor(user.search, q=str(symbol), tweet_mode='extended', lang='en').items(ct.num_of_tweets)

tweet_list = []
global_polarity = 0
for tweet in tweets:
    tw = tweet.full_text
    blob = TextBlob(tw)
    polarity = 0
    for sentence in blob.sentences:
        polarity += sentence.sentiment.polarity
        global_polarity += sentence.sentiment.polarity
    tweet_list.append(Tweet(tw, polarity))

global_polarity = global_polarity / len(tweet_list)

if df.iloc[-forecast_out-1]['Close'] < df.iloc[-1]['Prediction']:
    if global_polarity > 0:
        print("According to the predictions and twitter sentiment analysis -> Investing in %s is a GREAT idea!" % str(symbol))
    elif global_polarity < 0:
        print("According to the predictions and twitter sentiment analysis -> Investing in %s is a BAD idea!" % str(symbol))
else:
    print("According to the predictions and twitter sentiment analysis -> Investing in %s is a BAD idea!" % str(symbol))