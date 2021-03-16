# Data Mining (Text Mining)

# 1. Text Mining
Using the Coronavirus Tweets NLP data set from Kaggle https://www.kaggle.com/datatattle/covid-19-nlp-text-classification to predict the sentiment of Tweets relevant to Covid. The data set (Corona_NLP_test.csv file) contains 6 attributes.
<br>
Letter | Digit | 
------ | ------|
UserName| Anonymized attribute
ScreenName| Anonymized attribute
Location| Location of the person having made the tweet
TweetAt| Date
OriginalTweet| Textual content of the tweet
Sentiment| Emotion of the tweet
<br>

Because this is a quite big data set, use vectorized (e.g. pandas / numpy) built-in functions to effectively perform the various tasks. In this way, you will be able to run your code in few seconds. Otherwise, running your code might require a significant amount of time, e.g. in the case where for loops are used for accessing all elements of the data set.

## Task 1.1:
Compute the possible sentiments that a tweet may have, the second most popular sentiment in the tweets, and the date with the greatest number of extremely positive tweets. Convert the messages to lower case, replace non-alphabetical characters with whitespaces and ensure that the words of a message are separated by a single whitespace.

## Task 1.2:
Tokenize the tweets (i.e. convert each into a list of words), count the total number
of all words (including repetitions), the number of all distinct words and the 10 most frequent words in the corpus. Remove stop words, words with ≤ 2 characters and recalculate the number of all words (including repetitions) and the 10 most frequent words in the modified corpus. What do you observe?

## Task 1.3:
Plot a histogram with word frequencies, where the horizontal axis corresponds to
words, while the vertical axis indicates the fraction of documents in a which a word appears. The words should be sorted in increasing order of their frequencies. Because the size of the data set is quite big, use a line chart for this, instead of a histogram. In what way this plot can be useful for deciding the size of the term document matrix? How many terms would you add in a term-document matrix for this data set?

## Task 1.4:
Tokenize the tweets (i.e. convert each into a list of words), count the total number
of all words (including repetitions), the number of all distinct words and the 10 most frequent words in the corpus. Remove stop words, words with ≤ 2 characters and recalculate the number of all words (including repetitions) and the 10 most frequent words in the modified corpus. What do you observe?

# 2. Image Processing
Using the provided image data for performing image processing operations with skimage and scipy. The data set consists of the following 4 images:
File | Source | 
------ | ------|
avengers_imdb.jpg| https://www.imdb.com/
bush_house_wikipedia.jpg| https://en.wikipedia.org/ 
foestry_commission_gov_uk.jpg| https://www.gov.uk/ 
rolland_garros_tv5monde.jpg| http://www.tv5monde.com/ 
<br>

## Task 2.1: 
Determine the size of the avengers imdb.jpg image. Produce a grayscale and a blackand-white representation of it.

## Task 2.2:
Add Gaussian random noise in bush house wikipedia.jpg (with variance 0.1) and filter
the perturbed image with a Gaussian mask (sigma equal to 1) and a uniform smoothing mask (the latter of size 9x9).

## Task 2.3:
Divide forestry commission gov uk.jpg into 5 segments using k-means segmentation.

## Task 2.4:
Perform Canny edge detection and apply Hough transform on rolland garros tv5monde.jpg.






