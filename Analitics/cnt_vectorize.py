from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import pandas as pd

import nltk

nltk.download('stopwords')

stopwords_en = stopwords.words('english')

df = pd.read_csv("../Parse/datasets/final_datasets/total_critics.csv")

# df = df.loc[df['review'] != ""]
# df.to_csv("../Parse/datasets/final_datasets/total_critics.csv", encoding="utf-8", index=False)

df1 = df.loc[df['game_id'] == 1].reset_index()
del df1['index']

df1['review'] = df1['review'].astype(str).apply(nltk.word_tokenize)  # токенизация

stemmer = PorterStemmer()
df1['review'] = df1['review'].apply(
    lambda x: [stemmer.stem(word) for word in x])  # лемматизация

df1['review'] = df1['review'].apply(lambda x: " ".join(x))


count_vect = CountVectorizer(analyzer='word', stop_words=stopwords_en, strip_accents='unicode', ngram_range=(1, 2),
                             lowercase=True)

matrix = count_vect.fit_transform(df1['review'].astype(str)).toarray()
voc = count_vect.vocabulary_

sorted_keys = sorted(voc, key=voc.get)

df1_vect = pd.DataFrame(matrix, columns=sorted_keys)

df1_vect['Score'] = df1['score']


df1_vect.to_csv("../Parse/datasets/final_datasets/BG3_critics_vectorized.csv", encoding="utf-8", index=False)