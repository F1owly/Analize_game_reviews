import pandas as pd
import numpy as np
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from transformers import BertModel, BertTokenizer

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import nltk
import torch


class Preprocess(object):
    def __init__(self, path, game_id=None):
        self.path = path
        self.df = pd.read_csv(path)

        self.game_id = game_id
        if game_id is not None:
            self.df = self.df.loc[self.df['game_id'] == 1].reset_index()
            del self.df['index']

    def del_empty(self):
        self.df = self.df.loc[self.df['review'] != ""]
        self.df = self.df.dropna()

    def clear(self):
        def my_sub(x: str, s1: str, s2: str):
            return re.sub(s1, s2, x)

        self.df['review'] = self.df['review'].astype(str).apply(my_sub, args=("[^a-zA-Z]", " "))
        self.del_empty()

    def tokenize(self):
        self.clear()
        self.df['review'] = self.df['review'].astype(str).apply(nltk.word_tokenize)  # токенизация

    def steming(self):
        stemmer = PorterStemmer()
        self.df['review'] = self.df['review'].apply(
            lambda x: [stemmer.stem(word) for word in x])  # лемматизация

        self.df['review'] = self.df['review'].apply(lambda x: " ".join(x))

    def vectorize(self, ngram_range=None, stopwords_=None):
        self.tokenize()
        self.steming()

        if stopwords_ is None:
            nltk.download('stopwords')
            stopwords_ = stopwords.words('english')

        count_vect = CountVectorizer(analyzer='word', stop_words=stopwords_, strip_accents='unicode',
                                     ngram_range=(1, 2),
                                     lowercase=True)

        matrix = count_vect.fit_transform(self.df['review'].astype(str)).toarray()
        voc = count_vect.vocabulary_

        sorted_keys = sorted(voc, key=voc.get)
        scores = self.df['score']

        self.df = pd.DataFrame(matrix, columns=sorted_keys)

        self.df['Score'] = scores

    def tf_idf(self, ngram_range=(1, 1)):
        self.tokenize()
        self.steming()

        nltk.download('stopwords')
        stopwords_ = stopwords.words('english')

        vectorizer = TfidfVectorizer(stop_words=stopwords_, )
        matrix = vectorizer.fit_transform(self.df['review']).toarray()
        voc = vectorizer.vocabulary_

        sorted_keys = sorted(voc, key=voc.get)
        scores = self.df['score']

        self.df = pd.DataFrame(matrix, columns=sorted_keys)

        self.df['Score'] = scores

    def bert_embedings(self, type='all'):
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        inputs = tokenizer(list(self.df['review']), return_tensors='pt', truncation=True, padding='max_length',
                           max_length=256)

        with torch.no_grad():
            outputs = model(**inputs)

        if type == 'cls':
            self.df = outputs.last_hidden_state[:, 0, :]
        elif type == 'all':
            self.df = outputs.last_hidden_state
        else:
            raise ValueError()

    def save_csv(self, path):
        print(self.df)
        if isinstance(self.df, torch.Tensor):
            torch.save(self.df, path + '.pt')
        else:
            self.df.to_csv(path + ".csv", encoding="utf-8", index=False)


class Algorithm(object):
    def __init__(self, path: str):
        self.tensor = torch.load(path, weights_only=True).numpy()
        self.result = None

    def Kmeans(self, n_clusters=2):
        kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(self.tensor)
        self.result = kmeans.labels_
        return self.result

    def Reduce_dim(self, n_components=15):
        pca = PCA(n_components=15, random_state=42)
        self.tensor = pca.fit_transform(self.tensor)
        return self.result

def relevant_reviws(classes: np.ndarray, game_id = 1, path = "Parse/datasets/final_datasets/total_critics.csv"):
    df = pd.read_csv(path)
    df = df.loc[df['game_id'] == game_id]

    result = pd.DataFrame({})
    result['review'] = df['review']
    result['prior_mechanic'] = classes
    return result

embedings = Algorithm("Parse/datasets/final_datasets/bg3_critics_embeding_cls.pt")

classes = embedings.Kmeans(3)

print(relevant_reviws(classes))




