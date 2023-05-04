import torch
from joblib import load
import textwrap
import streamlit as st

device = 'cpu'

tokenizer = load('./pages/tokenizer.joblib')
model = load('./pages/model.joblib')
weights = model.load_state_dict(torch.load('./pages/model_weights.pt', map_location=device))

temperature = st.slider('Градус дичи:', min_value = 1., max_value = 20., value = 3.)
num_beams = st.slider('Число веток для поиска:', min_value = 1, max_value = 15, value = 7)
max_length = st.slider('Максимальная длина генерации:', min_value = 50, max_value = 150, value = 70)

prompt = st.text_input('Дайте волю фантазии!',)
if len(prompt) > 1:
    with torch.inference_mode():
        prompt = tokenizer.encode(prompt, return_tensors='pt').to(device)
        out = model.generate(
            input_ids=prompt,
            max_length=max_length,
            num_beams=num_beams,
            do_sample=True,
            temperature=temperature,
            top_k=50,
            top_p=0.6,
            no_repeat_ngram_size=3,
            num_return_sequences=3,
            ).cpu().numpy()
        for out_ in out:
            st.write(textwrap.fill(tokenizer.decode(out_), 40), end='\n------------------\n')
