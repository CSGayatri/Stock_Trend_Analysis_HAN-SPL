import tensorflow as tf 
import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Embedding, GRU, Dense, Input, Bidirectional, Dropout, LayerNormalization

# Load Pre-trained Word Embeddings (e.g., GloVe)
def load_pretrained_embeddings(word_index, embedding_dim=100):
    embeddings_index = {}
    with open("glove.6B.100d.txt", encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

class HybridAttentionHAN(Model):
    def __init__(self, vocab_size, embedding_dim, pre_trained_emb, hidden_size=256, dropout_rate=0.3):
        super(HybridAttentionHAN, self).__init__()

        self.embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim,
                                   weights=[pre_trained_emb], trainable=False, mask_zero=True)

        self.bi_gru = Bidirectional(GRU(hidden_size, return_sequences=True))
        self.layer_norm = LayerNormalization()
        self.dropout = Dropout(dropout_rate)

        # News-level attention
        self.news_attention = Dense(1, activation='tanh')

        # Temporal attention
        self.temp_attention = Dense(1, activation='tanh')

        # Output layer for three-class classification
        self.output_layer = Dense(3, activation='softmax')

    def call(self, x, training=False):
        x = self.embedding(x)  # Word Embedding
        x = self.bi_gru(x)  # Bi-GRU Encoding
        x = self.layer_norm(x)  # Normalize activations
        x = self.dropout(x, training=training)

        # News-Level Attention
        news_scores = self.news_attention(x)
        news_weights = tf.nn.softmax(news_scores, axis=1)
        news_representation = tf.reduce_sum(news_weights * x, axis=1)

        # Temporal Attention
        temp_scores = self.temp_attention(news_representation)
        temp_weights = tf.nn.softmax(temp_scores, axis=1)
        context_vector = tf.reduce_sum(temp_weights * news_representation, axis=1)

        return self.output_layer(context_vector)

class SelfPacedLearning:
    def __init__(self, lambda_init=0.01, lambda_step=1.1):
        self.lambda_value = lambda_init
        self.lambda_step = lambda_step

    def update_weights(self, losses):
        """ Update training sample weights based on loss values. """
        if np.isscalar(losses):  # Ensure losses is an array
            losses = np.array([losses])

        weights = np.where(losses < self.lambda_value, 1, 0)  # Weight = 1 if loss is below threshold
        self.lambda_value *= self.lambda_step  # Increase lambda over epochs

        return np.array(weights).reshape(-1)  # Ensure correct shape

# Training Configuration
vocab_size = 10000
embedding_dim = 100
word_index = {"stock": 1, "market": 2, "rise": 3}  # Replace with real tokenizer index
pre_trained_emb = load_pretrained_embeddings(word_index, embedding_dim)

# Initialize Model
model = HybridAttentionHAN(vocab_size, embedding_dim, pre_trained_emb)
model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), metrics=['accuracy'])

# Initialize SPL
spl = SelfPacedLearning()

# Train Model with SPL
for epoch in range(10):
    losses = model.fit(padded_sequences, labels, batch_size=16, epochs=1, verbose=1).history['loss']
    weights = spl.update_weights(losses)
    print(f"Epoch {epoch+1}: Updated SPL Weights - {weights}")
