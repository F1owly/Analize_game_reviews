import pandas as pd
import re

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from scipy.sparse import isspmatrix_csr, coo_array

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
    def __init__(self, path):


# bg3 = Preprocess("Parse/datasets/final_datasets/total_critics.csv", 1)
#
# bg3.bert_embedings(type='all')
# bg3.save_csv("Parse/datasets/final_datasets/bg3_critics_embeding_all")
# df = pd.read_csv("Parse/datasets/final_datasets/bg3_critics_tf_idf")
