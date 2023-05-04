from joblib import load

### pip install faiss-cpu
import faiss

### pip install datasets
from datasets import Dataset

import torch
import pandas as pd

import streamlit as st

device = 'cpu'

### подгрузка всех компонентов - модель, токенайзер и датасет с эмбеддингами
embeddings_dataset = load('./embeddings_dataset.joblib')
tokenizer = load('./tokenizer.joblib')
model = load('./model.joblib')

### функция возвращающая от БЕРТа только [CLS] опиывающий общий смысл всего предложения
def embed_bert_cls(text, model, tokenizer):
    t = tokenizer(text, padding=True, truncation=True, return_tensors='pt')
    with torch.no_grad():
        model_output = model(**{k: v.to(device) for k, v in t.items()})
    embeddings = model_output.last_hidden_state[:, 0, :]
    embeddings = torch.nn.functional.normalize(embeddings)
    return embeddings[0].cpu().numpy()

### функция ниже отдает готовый датасет с рекомендациями книг
def reccomend(input_string,n_neighbors=5):

    ### input_string - то, что вводит пользователь в аннотации, эмбеддинг пользовательского текста
    question_embedding = embed_bert_cls([input_string], model, tokenizer)

    ### n_neighbors - число предлагаемых системой книг, вводит пользователь, 
    ### поиск похожих книг по запросу
    scores, samples = embeddings_dataset.get_nearest_examples(
        "embeddings", question_embedding, k=n_neighbors
    )

    ### для корректной работы требуется формат таблиц huggingface, поэтому в конце 
    ### происходит перевод в пандас для удобства 
    samples_df = pd.DataFrame.from_dict(samples)
    samples_df["scores"] = scores
    samples_df.sort_values("scores", ascending=False, inplace=True)

    return samples_df

### конечный датасет: samples_df 


input = st.text_input('Your text here:', )
number = st.number_input('Insert a number', min_value = 1, max_value = 5, value = 3)

if len(input) > 1:
    st.write(reccomend(input, number))