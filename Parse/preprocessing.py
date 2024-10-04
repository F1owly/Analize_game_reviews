import pandas as pd
import pycld2 as cld2
import nltk
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt

nltk.download('punkt_tab')
nltk.download('stopwords')

from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

"""
Что же мы будем делать?
    1. Очистка текста
    2. Токенизация
    3. Удаление стоп-слов
    4. Стемминг
"""
total_critics_path = "../Parse/datasets/final_datasets/total_critics.csv"


def identify_lang(path: str):
    df = pd.read_csv(path)
    df['language'] = df['review'].astype(str).apply(lambda x: cld2.detect(x)[2][0][0])
    df.to_csv(path, encoding="utf-8", index=False)


# identify_lang("../Parse/datasets/final_datasets/total_critics.csv")

# for filename in os.listdir("../Parse/datasets/"):
#     if re.match(r".*\.csv$", filename):
#         identify_lang("../Parse/datasets/" + filename)


def clear_review(path: str):
    df = pd.read_csv(path)

    def my_sub(x: str, s1: str, s2: str):
        return re.sub(s1, s2, x)

    df['review'] = df['review'].astype(str).apply(my_sub, args=("[^a-zA-Z]", " "))
    df.to_csv(path, encoding="utf-8", index=False)


# clear_review(total_critics_path)


def tokenize_del_stop_words_stam_review(path: str):
    df = pd.read_csv(path)
    df['review'] = df['review'].astype(str).apply(nltk.word_tokenize)  # токенизация
    df['review'] = df['review'].apply(
        lambda x: [word for word in x if not word in stopwords.words('english')])  # удаление шума (стоп-слов)
    stemmer = PorterStemmer()
    df['review'] = df['review'].apply(
        lambda x: [stemmer.stem(word) for word in x])

    df.to_csv(path, encoding="utf-8", index=False)


tokenize_del_stop_words_stam_review(total_critics_path)

df = pd.read_csv(total_critics_path)

print(df['review'])
# plt.show()
