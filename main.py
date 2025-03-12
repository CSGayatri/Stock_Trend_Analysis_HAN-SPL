import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Attention, Input
from tensorflow.keras.models import Model

# Sample data (Replace with real stock news & labels)
texts = ["Stock market is volatile", "Tech stocks are rising", "Financial sector is strong"]
labels = np.array([0, 1, 2])  # 0: DOWN, 1: UP, 2: PRESERVE

# Tokenizer setup
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
MAX_SEQUENCE_LENGTH = 10
padded_sequences = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

# Placeholder for pre-trained word embeddings (Replace with real embeddings)
wordvec = np.random.rand(1000, 10)

# Define a simple LSTM model with Attention
def build_model(vocab_size=1000, embedding_dim=10, hidden_size=128):
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,))
    embedding = Embedding(vocab_size, embedding_dim, weights=[wordvec], trainable=False)(inputs)
    lstm_out = Bidirectional(LSTM(hidden_size, return_sequences=True))(embedding)
    
    # Simple Attention Layer
    attention = Attention()([lstm_out, lstm_out])
    lstm_out = tf.keras.layers.GlobalAveragePooling1D()(attention)

    output = Dense(3, activation='softmax')(lstm_out)
    model = Model(inputs, output)
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Build and train model
model = build_model()
model.fit(padded_sequences, labels, batch_size=2, epochs=5, verbose=1)

# Predict Example
sample_text = ["Tech stocks are booming"]
sample_seq = tokenizer.texts_to_sequences(sample_text)
sample_padded = pad_sequences(sample_seq, maxlen=MAX_SEQUENCE_LENGTH, padding='post')

prediction = model.predict(sample_padded)
print("Prediction:", prediction)
