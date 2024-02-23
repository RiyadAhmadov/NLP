# Import libraries
import re
import nltk 
import pickle
import numpy as np
import pandas as pd
from PIL import Image
import warnings as wg
wg.filterwarnings('ignore')
from textblob import TextBlob
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from textblob import Word , TextBlob
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')


data = pd.read_excel(r'C:\Users\HP\OneDrive\İş masası\NLP\NLP deploy\amazon.xlsx')

data['Review'] = data['Review'].apply(lambda x: str(x).lower())

# Replace newline characters with space
data['Review'] = data['Review'].replace('\n', ' ')

# Remove non-alphanumeric characters
data['Review'] = data['Review'].apply(lambda x: re.sub(r'[^a-zA-Z0-9 ]', '', str(x)))

# Remove digits
data['Review'] = data['Review'].str.replace('[\d]', '')

# Remove stopwords
sw = stopwords.words('english')
data['Review'] = data['Review'].apply(lambda x: ' '.join(word for word in x.split() if word not in sw))

# Lemmatization
lemmatizer = WordNetLemmatizer()
data['Review'] = data['Review'].apply(lambda x: ' '.join(lemmatizer.lemmatize(word) for word in x.split()))

# Let's extract rare words
word_count = pd.Series(' '.join(data['Review']).split()).value_counts()
rare = word_count[word_count <= 1]
data['Review'] = data['Review'].apply(lambda x: ' '.join(x for x in x.split() if x not in rare))


sia = SentimentIntensityAnalyzer()

data['Polarity'] = data['Review'].apply(lambda x: sia.polarity_scores(x)['compound'])
data['Polarity Label'] = data['Polarity'].apply(lambda x: 'Positive' if x > 0.1 else 'Negative')
data['Polarity Label'] = LabelEncoder().fit_transform(data['Polarity Label'])

X = data['Review']
y = data['Polarity Label']

train_x , test_x , train_y , test_y =  train_test_split(X ,y , random_state = 42)

# Let's convert input x to vector
tfidfVectorizer = TfidfVectorizer().fit(train_x)

x_train_tfidfWord = tfidfVectorizer.transform(train_x)
x_test_tfidfWord = tfidfVectorizer.transform(test_x)


# Let's create logistic regression model
log_model = LogisticRegression().fit(x_train_tfidfWord,train_y)


# Save the trained model
with open('nlp.pkl', 'wb') as model_file:
    pickle.dump(log_model, model_file)

# Save the TF-IDF vectorizer
with open('tfidfVectorizer.pkl', 'wb') as tfidf_file:
    pickle.dump(tfidfVectorizer, tfidf_file)