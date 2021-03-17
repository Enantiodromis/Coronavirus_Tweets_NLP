# IMPORTS
import numpy as np
import pandas as pd


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

# 1.2
# Tokenize the tweets (i.e. convert each into a list of words)
df['OriginalTweet_tokenized'] = df.apply(lambda row: row['OriginalTweet'].split(" "), axis = 1)
print(df['OriginalTweet_tokenized'])
# Counting the total number of all words (including repetitions)
df['len_tokens'] = df['OriginalTweet_tokenized'].apply(lambda x: len(x))
sum_of_len_tokens = df['len_tokens'].sum()
print("Total number of all words (including repetitions): " + str(sum_of_len_tokens))

# Counting the number of distinct words
df['unique_tokens'] = df['OriginalTweet_tokenized'].apply(lambda x: set(x))
df['len_tokens_unique'] = df['unique_tokens'].apply(lambda x: len((x)))
sum_of_len_tokens_unique = df['len_tokens_unique'].sum()
print("Total number of unique words: " + str(sum_of_len_tokens_unique))

# The 10 most frequent words in the corpus

# Remove stop words, words with ≤ 2 characters and recalculate the number of all words (including repetitions) and the 10 most frequent words in the modified corpus.