import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import numpy as np
import sys
import random
import fitz


with fitz.open("DOCUMENT_PATH") as doc:
    text = str()
    for page in doc:
        text += page.getText()
raw_text = text


# Should play with the filter argument in tokenizer, to get better text
tokenizer = Tokenizer(oov_token='<00V>', lower=True)
corpus = raw_text.split(".")  # Making a list of the texts, where there is a fullstop
tokenizer.fit_on_texts(corpus)  # Fitting the tokenizer on the corpus (takes the data and encode it)
# Dictionary containing key value pairs (key = word, value = token)
word_index = tokenizer.word_index
total_words = len(word_index) + 1  # Total number of unique words

# Processing the strings from the corpus to make it suitable for the network

input_sequences = list()

for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(len(token_list)):
        n_gram_sequences = token_list[:i+1]  # Creating n_gram sequences from every line
        input_sequences.append(n_gram_sequences)  # Keeping all the n_gram sequences in an list
# Lenght of the longest sequence, use it for padding
max_sequence_len = max([len(i) for i in input_sequences])

# Pad the n_gram sequences with the maximum seqnence length
# convert to also numpy array since we need to use it as input to the NN.
input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

# Take every input sequences token, and store everything except the last token
xs = input_sequences[:, :-1]
labels = input_sequences[:, -1]  # Take the last token

# Convert the last token into binary matrix (based on the total number of words)
ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)


model = Sequential([
    Embedding(total_words, 256, input_length=max_sequence_len-1),
    Bidirectional(LSTM(256, return_sequences=True)),
    Dropout(0.01),
    LSTM(256),
    Dense(total_words, activation='softmax')
])

optimizer = Adam(learning_rate=0.01)

model.compile(optimizer=optimizer, loss="categorical_crossentropy")


filepath = "best_model-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath,
                             monitor='loss',
                             verbose=1,
                             save_best_only=True,
                             mode='min')
callbacks_list = [checkpoint]

history = model.fit(xs, ys,
                    epochs=1,
                    batch_size=128,
                    verbose=1,
                    callbacks=callbacks_list)


# Generating Text

start_index = random.randint(0, len(raw_text)-120)
print(start_index)
seed_text = raw_text[start_index:start_index+100]
# seed_text = "Raskolnikov found himself lying on the floor" # Custom seed
next_words = 180
print(f"<START>{seed_text}<END>\n")
for i in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]  # Tokenzie input sequence
    token_list = pad_sequences([token_list], maxlen=max_sequence_len-1,
                               padding='pre')  # Padding the tokenized sequence
    predicted = model.predict(token_list, verbose=0)  # Predicting next word probability

    # Index from the probability of the model but with stochasticity
    next_index = np.random.choice(ys.shape[1], 1, p=predicted[0])[0]

    # From below is just printing the predicted word from the model.
    output_word = ""
    for word, index in tokenizer.word_index.items():  # Looping through the dictionary
        if index == next_index:
            output_word = word
            break
    sys.stdout.write(" " + output_word)  # Printing the output_word with spaces in between
    if i % 20 == 0:  # Since I did not train with "\n" token, therefore manual line break
        print("\n")
