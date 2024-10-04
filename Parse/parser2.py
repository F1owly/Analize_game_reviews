import os
import re
import pandas as pd

base_dir = "../Parse/datasets/"

games = ['ara_history_untold', 'baldur_gates3', 'elden_ring', 'epic_mickey_rebrushed', 'fifa25', 'reynatis',
         'the_legend_of_zelda', 'throne_and_liberty']


def first_prep(base_dir: str):
    for filename in os.listdir(base_dir):
        if not re.match(r".*\.csv$", filename):
            continue
        DataFrame = pd.read_csv(base_dir + filename)
        DataFrame = DataFrame.drop(['Unnamed: 0'], axis=1)
        DataFrame = DataFrame.reset_index()
        del DataFrame['index']
        DataFrame.to_csv(base_dir + filename, encoding="utf-8", index=False)


def norm_score(base_dir: str):
    for filename in os.listdir(base_dir):
        if not re.match(r".*\.csv$", filename):
            continue
        if re.match(r".*\/users\.csv$", filename):
            DataFrame = pd.read_csv(base_dir + filename)
            DataFrame['score'] = DataFrame['score'].apply(lambda x: x / 10)
        else:
            DataFrame = pd.read_csv(base_dir + filename)
            DataFrame['score'] = DataFrame['score'].apply(lambda x: x / 100)
        DataFrame.to_csv(base_dir + filename, encoding="utf-8", index=False)


def add_game_id(df: pd.DataFrame, id: int):
    df['game_id'] = [id] * len(df)
    return df


# first_prep(base_dir)

# norm_score(base_dir)
#
# game_id = 0
# for filename in os.listdir(base_dir):
#     if re.match(r".*_critics\.csv$", filename):
#         df = pd.read_csv(base_dir + filename)
#         df = add_game_id(df, game_id)
#         game_id += 1
#         df.to_csv(base_dir + filename, encoding="utf-8", index=False)

def norm_score_correction(base_dir: str):
    for filename in os.listdir(base_dir):
        if re.match(r".*_users\.csv$", filename):
            DataFrame = pd.read_csv(base_dir + filename)
            DataFrame['score'] = DataFrame['score'].apply(lambda x: x * 10)
            DataFrame.to_csv(base_dir + filename, encoding="utf-8", index=False)


# norm_score_correction(base_dir)

def total_critics_form(base_dir: str):
    for filename in os.listdir(base_dir):
        if re.match(r".*_critics\.csv$", filename):
            df = pd.read_csv(base_dir + filename)
            if os.path.exists(base_dir + "final_datasets/total_critics.csv"):
                df = pd.concat([df, pd.read_csv(base_dir + "final_datasets/total_critics.csv")])
                df.to_csv(base_dir + "final_datasets/total_critics.csv", encoding="utf-8", index=False)
            else:
                df.to_csv(base_dir + "final_datasets/total_critics.csv", encoding="utf-8", index=False)


total_critics_form(base_dir)
df = pd.read_csv("../Parse/datasets/final_datasets/total_critics.csv")
print(df.head(10))
