import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("../Parse/datasets/final_datasets/total_critics.csv")
sns.countplot(data=df, x='game_id')

plt.plot()
