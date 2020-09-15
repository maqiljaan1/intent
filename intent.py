import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import json
import matplotlib.pyplot as plt
import numpy as np

vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = 'post'
padding_type = 'post'
oov_tok = "<OOV>"
training_size = 20000
num_epochs = 500

with open("dataset//email_response.json","r") as f:
    datastore = json.load(f)

sentences = []
labels = []

for item in datastore:
    sentences.append(item['text_body'])
    labels.append(item['is_confirmed'])

training_size = int(len(sentences) * 0.6)

training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)

word_index = tokenizer.word_index

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.summary()

history = model.fit(training_padded, training_labels, epochs=num_epochs, validation_data=(testing_padded, testing_labels), verbose=2)


# def plot_graphs(history, string):
    # plt.plot(history.history[string])
    # plt.plot(history.history['val_'+string])
    # plt.xlabel("Epochs")
    # plt.ylabel(string)
    # plt.legend([string, 'val_'+string])
    # plt.show()

# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_sentence(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])

print(decode_sentence(training_padded[0]))
print(training_sentences[2])
print(labels[2])

model.save('saved_model/my_model')

e = model.layers[0]
weights = e.get_weights()[0]

# sentence = ["nah, let's do it next week", "please cancel this invitation", "maybe we should move it", "that sounds great"]
# sequences = tokenizer.texts_to_sequences(sentence)
# padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

# results = model.predict(padded)

def get_new_sentence(input_sentence):
    sentence = [input_sentence]
    sequences = tokenizer.texts_to_sequences(sentence)
    padded = pad_sequences(sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)
    results = model.predict(padded)
    i = 0
    for res in results:
        if res[0] > 0.5:
            print('"', sentence[i],'" - is a ','confirmation email ', res[0], sep='')
        else:
            print('"', sentence[i],'" - is a ','non-confirmation email ', res[0], sep='')
        i+=1

exit_phrase = 'quit'
stop_now = False

print('\n\n\nPlease enter a phrase below\n\n')

while not stop_now:
    user_input = input()
    if user_input == 'quit':
        stop_now = True
    get_new_sentence(user_input)
