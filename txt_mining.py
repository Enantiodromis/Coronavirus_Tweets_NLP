# IMPORTS
import numpy as np
import pandas as pd
import re

# Reading in the text_data csv file as a df (vectorised)
df = pd.read_csv('data/text_data/Corona_NLP_train.csv')

# 1.1
# Computing the possible sentiment values
unique_sentiment_values = df.Sentiment.unique()
print(unique_sentiment_values)

# The second most popular sentiment in the tweets
sentiment_values_count =  df.Sentiment.value_counts()
print("The second most popular sentiment in the tweets is: " + str(list(sentiment_values_count.items())[1][0]))

# Date with the greatest number of extremely positive tweets
max_ep_tweets_date = df[df.Sentiment == 'Extremely Positive']
max_ep_tweets_date_list = max_ep_tweets_date.groupby('TweetAt')['Sentiment'].value_counts().sort_values(ascending=False).reset_index(name = 'Counts')
print("Date of the max amount of extremely positive: " + max_ep_tweets_date_list['TweetAt'].iloc[0])

# Converting the messages to lower case
df = pd.DataFrame(df['OriginalTweet'].str.lower())

# Replace non-alphabetic characters with whitespaces and ensure that the words of a message are seperated by a single whitespace
df['OriginalTweet'] = df['OriginalTweet'].replace(to_replace = '[^a-z]+', value = ' ', regex = True).replace(to_replace = '[\s\s]+', value = ' ', regex = True)