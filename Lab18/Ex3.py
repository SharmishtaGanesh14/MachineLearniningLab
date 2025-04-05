# Required Libraries
# Before we begin, we supress deprecation warnings resulting from nltk on Kaggle
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
import pandas as pd
import re, nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import collections

nltk.download('punkt_tab')
# nltk.download('stopwords')
# nltk.download('wordnet')

# Sample CSV loading (replace this with your real dataset)
tweets = pd.read_csv("../datasets/Tweets.csv")  # Uncomment this for actual use

sentiment_counts = tweets.airline_sentiment.value_counts()
number_of_tweets = tweets.tweet_id.count()
print(sentiment_counts)

# Initialize Stopwords and Lemmatizer
stop_words = set(stopwords.words('english'))
wordnet_lemmatizer = WordNetLemmatizer()

# Normalization Function
def normalizer(tweet):
    only_letters = re.sub("[^a-zA-Z]", " ", tweet)  # Keep only letters
    tokens = nltk.word_tokenize(only_letters)[2:]   # Tokenize and skip first 2 tokens (@airline_name)
    lower_case = [l.lower() for l in tokens]
    filtered_result = list(filter(lambda l: l not in stop_words, lower_case))
    lemmas = [wordnet_lemmatizer.lemmatize(t) for t in filtered_result]
    return lemmas

# Apply normalization to tweets
pd.set_option('display.max_colwidth', None)  # Show full text in cells
tweets['normalized_tweet'] = tweets['text'].apply(normalizer)

# Function to create bigrams and trigrams
def ngrams(input_list):
    bigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:]))]
    trigrams = [' '.join(t) for t in list(zip(input_list, input_list[1:], input_list[2:]))]
    return bigrams + trigrams

# Apply n-gram creation
tweets['grams'] = tweets['normalized_tweet'].apply(ngrams)

# Function to count word frequencies
def count_words(input):
    cnt = collections.Counter()
    for row in input:
        for word in row:
            cnt[word] += 1
    return cnt

# Count top 20 n-grams in NEGATIVE tweets
neg_top_grams = tweets[tweets['airline_sentiment'] == 'negative'][['grams']].apply(count_words)['grams'].most_common(20)

# Count top 20 n-grams in POSITIVE tweets
pos_top_grams = tweets[tweets['airline_sentiment'] == 'positive'][['grams']].apply(count_words)['grams'].most_common(20)

# Print results
print("Top 20 n-grams in Negative Tweets:")
for gram, freq in neg_top_grams:
    print(f"{gram}: {freq}")

print("\nTop 20 n-grams in Positive Tweets:")
for gram, freq in pos_top_grams:
    print(f"{gram}: {freq}")


print(tweets)