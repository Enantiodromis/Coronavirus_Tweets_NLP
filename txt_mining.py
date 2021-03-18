# IMPORTS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

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

# Converting the messages to lower case replace non-alphabetic characters with whitespaces and ensure that the words of a message are seperated by a single whitespace
df.OriginalTweet = df.OriginalTweet.str.lower()\
                                         .replace(to_replace = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})+', value = '', regex = True)\
                                         .replace(to_replace = '[^a-z]+', value = ' ', regex = True)\
                                         .replace(to_replace = '[\s\s]+', value = ' ', regex = True)\

# 1.2
# Tokenize the tweets (i.e. convert each into a list of words) frequency of every word in the corpus
token_count = df.OriginalTweet.str.split(expand = True).stack().value_counts()

# Counting the total number of all words (including repetitions)
print("Total number of all words (including repetitions): " + str(sum(token_count)))

# Counting the number of distinct words
print("Total number of distinct words: " + str(len(token_count)))
# The 10 most frequent words in the corpus
print(token_count.head(10))

# Remove stop words, words with ≤ 2 characters and recalculate the number of all words (including repetitions) and the 10 most frequent words in the modified corpus.
df.OriginalTweet = df.OriginalTweet.apply(lambda x: [item for item in x.split() if item not in ENGLISH_STOP_WORDS and len(item) > 2]).apply(lambda x: ' '.join(x))

# Frequency of every word in the corpus
token_count = df.OriginalTweet.str.split(expand = True).stack().value_counts()

# Counting the total number of all words (including repetitions)
print("Total number of all words (including repetitions): " + str(sum(token_count)))
print(token_count.head(10))