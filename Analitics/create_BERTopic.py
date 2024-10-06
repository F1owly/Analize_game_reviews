from bertopic import BERTopic
import pandas as pd

df = pd.read_csv("../Parse/datasets/final_datasets/total_critics.csv")

df = df.loc[df['game_id'] == 1]

#~30c
mech_model = BERTopic()
mechs, probs = mech_model.fit_transform(df['review'])

mech_model.get_topic_info().to_csv("trash.csv")
print(pd.read_csv("trash.csv"))