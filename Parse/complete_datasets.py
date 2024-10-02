import pandas as pd

base_dir = 'datasets/'
games = ['ara_history_untold', 'baldur_gates3', 'elden_ring', 'epic_mickey_rebrushed', 'fifa25', 'reynatis',
         'the_legend_of_zelda', 'throne_and_liberty']

# for game in games:
#     if game in ['throne_and_liberty']:
#         df = pd.read_csv(base_dir + game + '_critics.csv')
#         df['score'] = df['score'].apply(lambda x: x / 100)
#         df.to_csv(base_dir + game + '_critics.csv', encoding='utf-8')
#         continue
#     df_users = pd.read_csv(base_dir + game + '_users.csv')
#     df_critics = pd.read_csv(base_dir + game + '_critics.csv')
#
#     df_users['score'] = df_users['score'].apply(lambda x: x / 10)
#     df_critics['score'] = df_critics['score'].apply(lambda x: x / 100)
#
#     df_users.to_csv(base_dir + game + '_users.csv', encoding='utf-8')
#     df_critics.to_csv(base_dir + game + '_critics.csv', encoding='utf-8')

for game in games:
    if game in ['throne_and_liberty']:
        df = pd.read_csv(base_dir + game + '_critics.csv')
        df.to_csv(base_dir + 'final_datasets/' + game + '.csv', encoding='utf-8')
        continue

    df = pd.concat([pd.read_csv(base_dir + game + '_critics.csv'), pd.read_csv(base_dir + game + '_users.csv')],
                   ignore_index=True)
    df = df.drop(['Unnamed: 0'], axis=1)
    df.to_csv(base_dir + 'final_datasets/' + game + '.csv', encoding='utf-8')

df_critics = pd.DataFrame({'review': [], 'score': [], 'game_id': []})
id = -1
for game in games:
    id += 1
    df_critics_new = pd.read_csv(base_dir + game + '_critics.csv')
    df_critics_new = df_critics_new.insert(1, 'game_id', [id]*len(df_critics_new))
    df_critics = pd.concat([df_critics, df_critics_new], ignore_index=True)
    print(df_critics_new)
    # df_critics = df_critics.drop(['Unnamed: 0'], axis=1)

df_users = pd.DataFrame({'review': [], 'score': [], 'game_id': []})
id = -1
for game in games:
    id+=1
    if game in ['throne_and_liberty']:
        continue
    df_users_new = pd.read_csv(base_dir + game + '_users.csv')
    df_users_new = df_users_new.insert(1, 'game_id', [id]*len(df_users_new))
    df_users = pd.concat([df_users, df_users_new], ignore_index=True)
    print(df_users_new)
    #df_users = df_users.drop(['Unnamed: 0'], axis=1)

df_critics.to_csv(base_dir + 'final_datasets/critics_reviews.csv', encoding='utf-8')
df_users.to_csv(base_dir + 'final_datasets/users_reviews.csv', encoding='utf-8')

df_total = pd.concat([df_users, df_critics], ignore_index=True)
df_total.to_csv(base_dir + 'final_datasets/total_reviews.csv', encoding='utf-8')
