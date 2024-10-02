import os
import re
import pandas as pd

base_dir = "../Parse/datasets/"


def first_prep(base_dir):
    for filename in os.listdir(base_dir):
        if not re.match(r".*\.csv$", filename):
            continue
        df = pd.read_csv(base_dir + filename)
        df = df.drop(['Unnamed: 0'], axis=1)
        df = df.reset_index()
        del df['index']
        df.to_csv(base_dir + filename, encoding="utf-8", index=False)


def norm_score(base_dir):
    for filename in os.listdir(base_dir):
        if not re.match(r".*\.csv$", filename):
            continue
        if re.match(r".*\/users\.csv$", filename):
            df = pd.read_csv(base_dir + filename)
            df['score'] = df['score'].apply(lambda x: x/10)
        else:
            df = pd.read_csv(base_dir + filename)
            df['score'] = df['score'].apply(lambda x: x / 100)
        df.to_csv(base_dir + filename, encoding="utf-8", index=False)



# first_prep(base_dir)
# norm_score(base_dir)

df = pd.read_csv("../Parse/datasets/baldur_gates3_critics.csv")
print(df.head(20))