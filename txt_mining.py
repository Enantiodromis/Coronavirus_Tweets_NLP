# IMPORTS
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from matplotlib.axis import Axis 
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset, InsetPosition
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn import metrics
from prettytable import PrettyTable

# Reading in the text_data csv file as a df (vectorised)
df = pd.read_csv('data/text_data/Corona_NLP_train.csv')

################
# Question 1.1 #
################

# Computing the possible sentiment values
unique_sentiment_values = df.Sentiment.unique()
#print(unique_sentiment_values)

# Computing the second most popular sentiment in the tweets  
sentiment_values_count =  df.Sentiment.value_counts()
second_popular_sentiment = list(sentiment_values_count.items())[1][0]
#print("The second most popular sentiment in the tweets is: " + second_popular_sentiment)

# Retrieving the date with the greatest number of extremely positive tweets
max_ep_tweets_date = df[df.Sentiment == 'Extremely Positive']
max_ep_tweets_date_list = max_ep_tweets_date.groupby('TweetAt')['Sentiment'].value_counts().sort_values(ascending=False).reset_index(name = 'Counts')
date_of_max_ep_tweets = max_ep_tweets_date_list['TweetAt'].iloc[0]
#print("Date of the max amount of extremely positive: " + date_of_max_ep_tweets)

# Converting the messages to lower case replace non-alphabetic characters with whitespaces and ensuring that the words of a message are seperated by a single whitespace
df['OriginalTweet_modified'] = df.OriginalTweet.str.lower()\
                                         .replace(to_replace = '(https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|www\.[a-zA-Z0-9][a-zA-Z0-9-]+[a-zA-Z0-9]\.[^\s]{2,}|https?:\/\/(?:www\.|(?!www))[a-zA-Z0-9]+\.[^\s]{2,}|www\.[a-zA-Z0-9]+\.[^\s]{2,})+', value = '', regex = True)\
                                         .replace(to_replace = '[^a-z]+', value = ' ', regex = True)\
                                         .replace(to_replace = '[\s\s]+', value = ' ', regex = True)\

################
# Question 1.2 #
################

# Tokenizing the tweets (i.e. convert each into a list of words) frequency of every word in the corpus
token_count = df.OriginalTweet_modified.str.split(expand = True).stack().value_counts()

# Counting the total number of all words (including repetitions) and distinct words:
def word_count(input_series):
    all_total, distinct_total = str(sum(input_series)), str(len(input_series))
    return all_total, distinct_total
all_words, distinct_words = word_count(token_count)

# Retrieving the top ten most frequently used words
top_ten = [word for word in token_count[:10].keys()]

# Removing stop words, words with ≤ 2 characters and recalculating the number of all words (including repetitions) and the 10 most frequent words in the modified corpus.
df.OriginalTweet_modified = df.OriginalTweet_modified.apply(lambda x: [item for item in x.split() if item not in ENGLISH_STOP_WORDS and len(item) > 2]).apply(lambda x: ' '.join(x))

# Frequency of every word in the corpus after stop words and words with ≤ 2 characters have been removed
token_count_after = df.OriginalTweet_modified.str.split(expand = True).stack().value_counts()
all_words_after, distinct_words_after = word_count(token_count_after)

# Retrieving the top ten most frequently used words
top_ten_after = [word for word in token_count_after[:10].keys()]

################
# Question 1.3 #
################

# Plotting line chart with word frequencies, where the horizontal axis corresponds to words, while the vertical axis indicates the fraction of documents in which a word appears. 
# The words are sorted in increasing order of their frequencies.
document = df.OriginalTweet_modified.tolist()

vectorizer =  CountVectorizer()
vector = vectorizer.fit_transform(document) 
  
# Manipulating the returned data to produce data for plotting.
vector_summed = sum(vector.toarray() > 0).tolist()
words_and_sums = sorted(zip(vector_summed, vectorizer.get_feature_names()), key= lambda x : x[0])

ordered_words_total = [x[1] for x in words_and_sums]
ordered_frequencies_total = [x[0]/len(df.OriginalTweet) for x in words_and_sums]

ordered_words_100 = [x[1] for x in words_and_sums[-100:]]
ordered_frequencies_100 = [x[0]/len(df.OriginalTweet) for x in words_and_sums[-100:]]

ordered_words_total_len = len(ordered_frequencies_total)
x_range_total = range(ordered_words_total_len)

# Plotting
plt.figure(figsize=(10,6))
plt.plot(x_range_total,ordered_frequencies_total)
plt.xticks(np.arange(0, ordered_words_total_len, 3000))
plt.xticks(rotation = 50)
plt.tick_params(axis='x', which='major', labelsize=8)
plt.title('Word Frequencies (entire dataset)')
plt.xlabel('Words (word index)')
plt.ylabel('Frequencies (fraction of documents in which the word appears)')
plt.savefig('outputs/word_frequency.png')

# Plotting
plt.figure(figsize=(10,6))
plt.plot(ordered_words_100,ordered_frequencies_100)
plt.xticks(rotation = 90)
plt.tick_params(axis='x', which='major', labelsize=6)
plt.title('Word Frequencies (top 100 word frequencies)')
plt.xlabel('Words')
plt.ylabel('Frequencies (fraction of documents in which the word appears)')
plt.savefig('outputs/word_frequency_100.png')

################
# Question 1.4 #
################

# Producing a Multinomial Naive Bayes classiﬁer for the Coronavirus Tweets NLP data set using scikit-learn. 
X = np.array(df.OriginalTweet_modified)
y = np.array(df.Sentiment)

X = vectorizer.fit_transform(X)

# MultinomialNB
nb = MultinomialNB()
nb.fit(X, y)

# Class predictions for X_train_tweet
y_pred_class = nb.predict(X)

# Calculate accuracy and error rate of class predictions
accuracy = metrics.accuracy_score(y, y_pred_class)
error_rate = str(round((1 - accuracy) * 100,2)) + "%"


#######################################
# FORMATTING OUTPUT USING PRETTYTABLE #
#######################################

# OUTPUT FOR 1.1
table_1 = PrettyTable()
table_1.title = 'Text Mining 1.1'
table_1.field_names = ["Question", "Findings"]
table_1.add_row(['Possible sentiments of a tweet: ', unique_sentiment_values])
table_1.add_row(['Second most popular sentiment: ', second_popular_sentiment])
table_1.add_row(['Date with greatest extremely positive tweets: ', date_of_max_ep_tweets])
table_1.align = "l"
print(table_1)

# OUTPUT FOR 1.2
table_2 = PrettyTable()
table_2.title = 'Text Mining 1.2'
table_2.field_names = ["Question", "Findings"]
table_2.add_row(['Total number of all words (including repetitions):',  all_words])
table_2.add_row(['Total number of all distinct words:',  distinct_words])
table_2.add_row(['10 most frequent words in the corpus:', top_ten])
table_2.add_row(['Total number of all words (including repetitions) (After stop words and words with <= 2 characters have been removed):',  all_words_after])
table_2.add_row(['10 most frequent words in the corpus (After stop words and words with <= 2 characters have been removed):', top_ten_after])
table_2.align = "l"
print(table_2)

# OUTPUT FOR 1.4
table_3 = PrettyTable()
table_3.title = 'Text Mining 1.3'
table_3.field_names = ["Question", "Findings"]
table_3.add_row(["Multinomial Naive Bayes classifier error rate:", error_rate])
table_2.align = "l"
print(table_3)