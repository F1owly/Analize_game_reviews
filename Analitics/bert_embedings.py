import pandas as pd
from transformers import BertModel, BertTokenizer
import torch
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

df = pd.read_csv("../Parse/datasets/final_datasets/bg3_critics_stemmed.csv")

inputs = tokenizer(list(df['review']), return_tensors='pt', truncation=True, padding='max_length', max_length=256)

with torch.no_grad():
    outputs = model(**inputs)

# Эмбеддинг CLS-токена
cls_embedding = outputs.last_hidden_state[:, 0, :]

# Эмбеддинги всех токенов
all_token_embeddings = outputs.last_hidden_state

print("CLS Embedding shape:", cls_embedding.shape)
print("All tokens Embedding shape:", all_token_embeddings.shape)