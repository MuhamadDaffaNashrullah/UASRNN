# train_model.py

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Path dan direktori
DATA_PATH = "ulasan_digitalent_labeled.csv"
WORDCLOUD_DIR = "static/wordcloud"
CHART_DIR = "static/charts"
MODEL_PATH = "model/rnn_model.h5"

# Buat folder yang dibutuhkan
os.makedirs(WORDCLOUD_DIR, exist_ok=True)
os.makedirs(CHART_DIR, exist_ok=True)
os.makedirs("model", exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
df = df[['cleaned_ulasan', 'label']].dropna()

# Tokenisasi
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(df['cleaned_ulasan'])
X = tokenizer.texts_to_sequences(df['cleaned_ulasan'])
X = pad_sequences(X, maxlen=100)

y = to_categorical(df['label'], num_classes=3)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model RNN
model = Sequential([
    Embedding(input_dim=5000, output_dim=64, input_length=100),
    LSTM(64, return_sequences=False),
    Dense(3, activation='softmax')
])
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=1)

# Simpan model
model.save(MODEL_PATH)

# ==========================
# Visualisasi
# ==========================

# WordCloud per label
label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}

for label_id, label_name in label_map.items():
    text = ' '.join(df[df['label'] == label_id]['cleaned_ulasan'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    wc.to_file(f"{WORDCLOUD_DIR}/{label_name}.png")

# Bar Chart
label_counts = df['label'].value_counts().sort_index()
label_counts.index = label_counts.index.map(label_map)

plt.figure(figsize=(6, 4))
label_counts.plot(kind='bar', color=['red', 'gray', 'green'])
plt.title("Distribusi Label Sentimen")
plt.xlabel("Sentimen")
plt.ylabel("Jumlah")
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/bar_chart.png")
plt.close()

# Pie Chart
plt.figure(figsize=(6, 6))
label_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, colors=['red', 'gray', 'green'])
plt.ylabel('')
plt.tight_layout()
plt.savefig(f"{CHART_DIR}/pie_chart.png")
plt.close()

print("Model dan visualisasi berhasil dibuat.")

# Setelah model dan tokenizer disimpan

# ==========================
# WordCloud per label
# ==========================

label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
wordcloud_dir = 'static/wordcloud'
os.makedirs(wordcloud_dir, exist_ok=True)

for label_id, label_name in label_map.items():
    text = ' '.join(df[df['label'] == label_id]['cleaned_ulasan'])
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    wc.to_file(f"{wordcloud_dir}/{label_name}.png")

print("Wordclouds untuk setiap sentimen berhasil dibuat.")
