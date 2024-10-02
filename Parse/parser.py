import re
import pandas as pd
import os.path

base_dir = "../Parse/"
path_txt = "source/ara_history_untoldPC_users.txt"
path_csv = "datasets/ara_history_untold_users.csv"

key_word1 = '<span data-v-e408cafe="">'
pattern1 = r'<span data-v-e408cafe="">(\d+)</span>'

key_word2 = '<div class="c-siteReview_quote g-outer-spacing-bottom-medium"><span>'
pattern2 = r'<div class="c-siteReview_quote g-outer-spacing-bottom-medium"><span>(.*?)</span>'

scores = []
reviews = []
with open(base_dir + path_txt, encoding='utf-8') as file:
    for line in file:
        if key_word1 in line:
            match = re.search(pattern1, line)
            if match:
                scores.append(int(match.group(1)))
            else:
                scores.append(-1)

        if key_word2 in line:
            match = re.search(pattern2, line)
            if match:
                reviews.append(match.group(1))
            else:
                reviews.append("None")

df_base = pd.DataFrame({'review': reviews, 'score': scores})

if os.path.exists(base_dir + path_csv):
    df = pd.read_csv(base_dir + path_csv)
    df_base = pd.concat([df_base, df])
    df_base = df_base.drop(["Unnamed: 0"], axis = 1)

df_base.to_csv(base_dir + path_csv)
print(path_txt)
print(path_csv)
