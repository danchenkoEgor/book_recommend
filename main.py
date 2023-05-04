import streamlit as st
st.set_page_config(page_title="FindMyBook", page_icon="📚", menu_items=None, initial_sidebar_state="auto")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import csv
from joblib import load
from transformers import AutoTokenizer, AutoModel
from faiss_file import model, tokenizer, embeddings_dataset, embed_bert_cls, recommend


# Модель, токенайзер, датасет, kmeans, функция рекомендаций

device = 'cpu'

tokenizer_k = AutoTokenizer.from_pretrained("cointegrated/rubert-tiny")
model_k = AutoModel.from_pretrained("cointegrated/rubert-tiny")
kmeans = load('kmeans.joblib')
emb = load('final_emb.joblib')

def recomendation(input):

    user_input = embed_bert_cls(input, model_k, tokenizer_k)
    label = kmeans.predict(user_input.reshape(1, -1))[0]
    sample_df = emb[emb['labels'] == label].copy()
    sample_df['cosine'] = sample_df['embeddings'].apply(lambda x: np.dot(x, user_input) / (np.linalg.norm(x) * np.linalg.norm(user_input)))

    return sample_df.sort_values('cosine', ascending=False)



st.title('Умный поиск книг')

with st.sidebar:
        st.markdown('Добро пожаловать в мир **FindMyBook** - самого умного поисковика книг! Это как твой личный библиотекарь, который знает все о тебе и твоих предпочтениях в литературе! Это не просто обычный поисковик, который ищет книги по авторам или названиям, это настоящий литературный детектив, который проникает в глубь содержания книг и помогает найти именно те, которые оставят неизгладимое впечатление.')
        st.markdown('**FindMyBook** работает на основе передовых алгоритмов искусственного интеллекта, которые позволяют ему анализировать содержание книг и находить связи между ними. Этот поисковик сможет найти книгу, которая понравится именно тебе, учитывая твои предпочтения и интересы.')
        st.markdown('Не нужно тратить время на бесконечный поиск книг в огромных онлайн-библиотеках. Просто введи тему, которая тебя интересует, и **FindMyBook** уже начнет искать книги, которые подходят именно тебе!')
        
user_prompt = st.text_area(label='Введите запрос:', placeholder="Хочу прочитать о...", height=200)

books_per_page = st.number_input('Количество рекомендаций:', min_value=1, max_value=5, value=3)

button = st.button("Найти")
        
tab1, tab2 = st.tabs(["Faiss Search", "K-Mean"])

with tab1:

    if button and len(user_prompt) > 1:
        book_recs = recommend(user_prompt, books_per_page)
        for i in range(books_per_page):

            col1, col2 = st.columns([2,7])
            with col1:
                image = book_recs['image_url'].iloc[i]
                st.image(image)

            with col2:
                title = book_recs['title'].iloc[i]
                try:
                    author = book_recs['author'].iloc[i].rstrip()
                except:
                    author = book_recs['author'].iloc[i]
                annotation = book_recs['annotation'].iloc[i]
                st.subheader(title)
                st.markdown(f'_{author}_')
                st.caption(annotation)
                st.markdown(f"[Подробнее...]({book_recs['page_url'].iloc[i]})")
                st.divider()

with tab2:

    book_recs = recommend(user_prompt, books_per_page)
    if button and len(user_prompt) > 1:
        
        book_recs = recomendation(user_prompt)
        for i in range(books_per_page):

            col1, col2 = st.columns([2,7])
            with col1:
                image = book_recs['image_url'].iloc[i]
                st.image(image)

            with col2:
                title = book_recs['title'].iloc[i]
                try:
                    author = book_recs['author'].iloc[i].rstrip()
                except:
                    author = book_recs['author'].iloc[i]
                annotation = book_recs['annotation'].iloc[i]
                st.subheader(title)
                st.markdown(f'_{author}_')
                st.caption(annotation)
                st.markdown(f"[Подробнее...]({book_recs['page_url'].iloc[i]})")
                st.divider()
