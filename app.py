from flask import Flask, render_template, request
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
import os

app = Flask(__name__)

# Load model
MODEL_PATH = "model/rnn_model.h5"
model = load_model(MODEL_PATH)

# Load data & tokenizer ulang agar konsisten
df = pd.read_csv("ulasan_digitalent_labeled.csv")
df = df[['cleaned_ulasan', 'label']].dropna()

tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_ulasan'])

max_len = 100
label_map = {0: 'Negatif', 1: 'Netral', 2: 'Positif'}

def predict_sentiment(text):
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=max_len)
    pred = model.predict(padded)[0]
    label_index = np.argmax(pred)
    return label_map[label_index], float(pred[label_index])

@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    confidence = None

    if request.method == 'POST':
        user_input = request.form['ulasan']
        if user_input.strip():
            result, confidence = predict_sentiment(user_input)

    return render_template("index.html", prediction=result, confidence=confidence)

if __name__ == '__main__':
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
