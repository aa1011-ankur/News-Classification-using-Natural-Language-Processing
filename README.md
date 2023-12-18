# News-Classification-using-Natural-Language-Processing
The project is basically AI news classification using Natural Language  Processing which includes the concepts like supervised learning.

# imports libs
import nltk
import pandas as pd
nltk.download('punkt')
fake = pd.read_csv("Fake.csv")
genuine = pd.read_csv("True.csv")
fake
genuine
fake["genuineness"] = 0
genuine["genuineness"] = 1
data = pd.concat([fake,genuine], axis=0)
data
data = data.reset_index(drop=True)
data = data.drop(['title','subject','date'], axis=1)
# Tokenization
from nltk.tokenize import word_tokenize
data['text'] = data['text'].apply(word_tokenize)
# Stemming
from nltk.stem.snowball import SnowballStemmer
sb = SnowballStemmer("english", ignore_stopwords=False)
def stem_it(text):
 return[sb.stem(word) for word in text]
data['text'] = data['text'].apply(stem_it)
# Stop words removal
def stopword_remover(text):
 return[word for word in text if len(word>>2)]
data['text'] = data['text'].apply(' '.join)
data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['text'], 
data['genuineness'], test_size=0.25)
X_train
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(max_df=0.7)
tfidf_train = tfidf.fit_transform(X_train)
tfidf_test = tfidf.transform(X_test)
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression(max_iter=900)
model1.fit(tfidf_train, y_train)
pred1 = model1.predict(tfidf_test)
pred1
y_test
from sklearn.metrics import accuracy_score
cr1 = accuracy_score(y_test, pred1)
cr1
from sklearn.linear_model import PassiveAggressiveClassifier
model2 = PassiveAggressiveClassifier(max_iter=100)
model2.fit(tfidf_train, y_train)
pred2 = model2.predict(tfidf_test)
cr2 = accuracy_score(y_test, pred2)
cr2
