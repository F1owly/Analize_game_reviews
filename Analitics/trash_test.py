import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

import nltk

import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../Parse/datasets/final_datasets/bg3_critics_stemmed.csv")

nltk.download('stopwords')
stopwords_ = stopwords.words('english')

vectorizer = TfidfVectorizer(stop_words=stopwords_)
matrix = vectorizer.fit_transform(df['review']).toarray()
voc = vectorizer.vocabulary_

sorted_keys = sorted(voc, key=voc.get)
scores = df['score']

df = pd.DataFrame(matrix, columns=sorted_keys)

df['Score'] = scores

print(df)
