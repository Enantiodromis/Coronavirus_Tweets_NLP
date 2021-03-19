# IMPORTS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, InsetPosition


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

# Counting the total number of all words (including repetitions) and distinct words:
def word_count(input_series):
    all_total, distinct_total = str(sum(token_count)), str(len(token_count))
    return all_total, distinct_total

all_words, distinct_words = word_count(token_count)
print("Number of all words (including repetitions): %s" %all_words)
print("Number of distinct words: %s" %distinct_words)
# The 10 most frequent words in the corpus
print(token_count.head(10))

# Remove stop words, words with ≤ 2 characters and recalculate the number of all words (including repetitions) and the 10 most frequent words in the modified corpus.
df.OriginalTweet = df.OriginalTweet.apply(lambda x: [item for item in x.split() if item not in ENGLISH_STOP_WORDS and len(item) > 2]).apply(lambda x: ' '.join(x))

# Frequency of every word in the corpus
token_count = df.OriginalTweet.str.split(expand = True).stack().value_counts()
all_words, distinct_words = word_count(token_count)

# Counting the total number of all words (including repetitions)
print("Number of all words (including repetitions): %s" %all_words)
print(token_count.head(10))

# 1.3
# Plot a histogram with word frequencies, where the horizontal axis corresponds to words, while the vertical axis indicates the fraction of documents in a which a word appears. 
# The words should be sorted in increasing order of their frequencies. Use a line chart for this, instead of a histogram.
document = df.OriginalTweet.tolist()
#print(document)
vectorizer =  CountVectorizer() # Creating a vectorizer object
vectorizer.fit(document)

# Encode the Document 
vector = vectorizer.transform(document) 
  
# Summarizing the Encoded Texts 
vector_summed = sum(vector.toarray() > 0).tolist()
words_and_sums = sorted(zip(vector_summed, vectorizer.get_feature_names()), key= lambda x : x[0])

ordered_words = [x[1] for x in words_and_sums]
ordered_frequencies = [x[0]/len(df.OriginalTweet) for x in words_and_sums]

plt.figure(figsize=(10,6))
plt.plot(ordered_words,ordered_frequencies)
plt.xticks([])   

plt.title('Word Frequencies')
plt.xlabel('Words')
plt.ylabel('Frequencies (Fraction of documents in a which the word appears)')
plt.show()
