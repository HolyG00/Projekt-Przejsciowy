import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.metrics import Precision, Recall, F1Score
import pickle

# Wczytywanie danych
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# Połączenie tytułów i treści artykułów w obu zbiorach danych
train_data['combined_text'] = train_data[train_data.columns[1]] + " " + train_data[train_data.columns[2]]
test_data['combined_text'] = test_data[test_data.columns[1]] + " " + test_data[test_data.columns[2]]

# Ekstrakcja etykiet i tekstów do osobnych list
train_texts = train_data['combined_text'].tolist()
train_labels = train_data[train_data.columns[0]].tolist()

test_texts = test_data['combined_text'].tolist()
test_labels = test_data[test_data.columns[0]].tolist()

# Parametry przetwarzania tekstu
max_words = 10000  # Liczba słów w słowniku
max_len = 200      # Maksymalna długość każdego artykułu

# Tokenizacja i konwersja tekstów na sekwencje
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(train_texts)
train_sequences = tokenizer.texts_to_sequences(train_texts)
test_sequences = tokenizer.texts_to_sequences(test_texts)

# Zapisanie tokenizera do pliku
with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Wyrównanie sekwencji
train_padded = pad_sequences(train_sequences, maxlen=max_len)
test_padded = pad_sequences(test_sequences, maxlen=max_len)

# Konwersja etykiet na format kategoryczny
num_classes = len(train_data[train_data.columns[0]].unique())
train_labels_cat = to_categorical([label - 1 for label in train_labels], num_classes=num_classes)
test_labels_cat = to_categorical([label - 1 for label in test_labels], num_classes=num_classes)

# Budowanie modelu sieci neuronowej
model = Sequential()
model.add(Embedding(max_words, 120, input_length=max_len))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dropout(0.5))  
model.add(Dense(num_classes, activation='softmax', kernel_regularizer=l2(0.01))) 

# Kompilacja modelu
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', Precision(), Recall(), F1Score()])

# Trenowanie modelu
model.fit(train_padded, train_labels_cat, batch_size=256, epochs=4, validation_split=0.2)
model.save('model_klas_words.h5')  

# Ewaluacja modelu na danych testowych
loss, accuracy, precision, recall, f1 = model.evaluate(test_padded, test_labels_cat)
print(f"Loss: {loss}, Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, f1{f1}") 
