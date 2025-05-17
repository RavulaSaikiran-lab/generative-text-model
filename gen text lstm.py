import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Sample dataset (expand for better results)
text_corpus = """Artificial intelligence is transforming industries globally. 
It enhances automation, improves efficiency, and offers powerful analytical insights. 
AI's impact is seen in healthcare, finance, and education. Its future holds endless possibilities."""

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts([text_corpus])
word_index = tokenizer.word_index
total_words = len(word_index) + 1

# Prepare sequences for training
input_sequences = []
for line in text_corpus.split("."):
    tokens = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(tokens)):
        input_sequences.append(tokens[:i+1])

# Pad sequences
max_length = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_length, padding='pre')

# Split data into input (X) and output (y)
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define LSTM model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(total_words, 100, input_length=max_length-1),
    tf.keras.layers.LSTM(200, return_sequences=True),
    tf.keras.layers.LSTM(200),
    tf.keras.layers.Dense(total_words, activation='softmax')
])

# Compile and train the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=150, verbose=1)

# Function to generate coherent text with a minimum word count
def generate_lstm_text(seed_text, min_words=30):
    generated_text = seed_text
    while len(generated_text.split()) < min_words:
        sequence = tokenizer.texts_to_sequences([generated_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length-1, padding='pre')
        predicted = np.argmax(model.predict(sequence), axis=-1)
        output_word = next((word for word, index in word_index.items() if index == predicted), "")
        generated_text += " " + output_word if output_word else ""
    return generated_text

# Example usage
topic = "Artificial intelligence"
generated_paragraph = generate_lstm_text(topic)
print(generated_paragraph)
