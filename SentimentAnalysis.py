from warnings import filterwarnings
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from nltk.corpus import stopwords
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder
from textblob import Word, TextBlob
from wordcloud import WordCloud

import nltk
nltk.download("stopwords")
nltk.download("punkt")
nltk.download("wordnet")

import matplotlib.pyplot as plt
import pandas as pd
from google.colab import files
import io


uploaded = files.upload()
df= pd.read_excel(io.BytesIO(uploaded['All_Reviews_PH.xlsx']))

pd.set_option("display.max_rows",100)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_colwidth",None)

df_review=df

print(df_review.head())
print(df_review.shape)

"""^[\w\s]=remove all except alphanumeric and whitespaces tabs and spaces [\d] remove digits"""

df_review['Review']=df_review['Review'].str.replace('\n', ' ')
df_review['Review']=df_review['Review'].str.replace('[^\w\s]', '')
print(df_review.head(100))

"""Print the rows with Missing NAN values"""

print(df_review[df_review.isna().any(axis=1)])

print(df_review.head())

"""HOW MANY POSITIVE REVIEWS, HOW MANY NEGATIVE REVIEWS"""

print(df_review[df_review['Sentiment']==1].count())
print(df_review[df_review['Sentiment']==0].count())

"""Converting datatype of a column, here tried to convert float type of Sentiment to int"""

#print(df_review['Sentiment'].astype(int))

"""Stopwords"""

from nltk.corpus import stopwords
sw=stopwords.words("english")
print(sw)
words=['no','not']
sw.remove('not')
sw.remove('no')
print(sw)

df_review['Review']=df_review['Review'].apply(lambda x:" ".join(x for x in str(x).split() if x not in sw))
print(df_review.head())

"""TOKENIZATION"""

w_tokenizer=nltk.tokenize.WhitespaceTokenizer()
lemmatizer=nltk.stem.WordNetLemmatizer()

def lemmatize_text(text):
  return [lemmatizer.lemmatize(w,pos='v') for w in w_tokenizer.tokenize(text)]

df_review['Review']=df_review.Review.apply(lemmatize_text)
print(df_review.head())

"""Lemmatization"""

df_review['NewReview'] = df_review['Review'].apply(lambda s1: ' '.join(map(str,s1)) )

print(df_review['NewReview'].head())

df_review['NewReview']=df_review['NewReview'].str.lower()
df_review.head()

"""Sentiemnt Analysis"""

from sklearn.model_selection import train_test_split

train_data,test_data=train_test_split(df_review,test_size=0.2,random_state=25)

print(test_data.shape)
print(test_data[test_data['Sentiment']==1].count())

X_train=train_data['NewReview']
y_train=train_data['Sentiment']
X_test=test_data['NewReview']
y_test=test_data['Sentiment']

print(X_train.head())
print(X_test.head())

"""Vectorizer"""

from sklearn.feature_extraction.text import TfidfVectorizer

train_vectorizer=TfidfVectorizer(use_idf=True,lowercase=True,strip_accents='ascii')
train_review_tfidf=train_vectorizer.fit_transform(X_train)

print(train_review_tfidf.shape)
type(train_review_tfidf)

"""printing first 5 elements and other is about ngrams"""

#print(review_tfidf.toarray()[1:5])
#train_vectorizer_ngram=TfidfVectorizer(ngram_range=(2,3))
#review_tfidf_ngram=train_vectorizer_ngram.fit_transform(X)

print(y_train.head())
print(y_train.shape)
print(train_review_tfidf.shape)

"""LOGISTIC REGRESSION"""

log_model=LogisticRegression().fit(train_review_tfidf, y_train)

cross_val_score(log_model,train_review_tfidf,y_train,scoring='accuracy',cv=5).mean()

"""TESTING"""

nr=train_vectorizer.transform(X_test)
log_model.predict(nr)

cross_val_score(log_model,nr,y_test,scoring='accuracy',cv=5).mean()

predicted=log_model.predict(nr)
actual=y_test

#predicted=pd.DataFrame(predicted)
print(predicted)

pd.set_option("display.max_rows",None)

fin_df=pd.DataFrame()
fin_df['Actual']=actual
fin_df['Predicted']=predicted
print(fin_df)

print(fin_df)

fin_df['Reviews']=X_test

print(fin_df.head(20))

fin_df.to_csv('result.csv')
