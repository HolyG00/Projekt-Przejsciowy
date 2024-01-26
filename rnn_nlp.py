# Importy z word_infinitive.py
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pickle
import pandas as pd
import random

# Importy z klas_words.py
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Funkcja do odczytu tekstu z pliku
def read_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Wczytywanie danych
test_data = pd.read_csv('test.csv')

# Połączenie tytułów i treści artykułów w obu zbiorach danych
test_data['combined_text'] = test_data[test_data.columns[1]] + " " + test_data[test_data.columns[2]]
test_texts = test_data['combined_text'].tolist()
test_labels = test_data[test_data.columns[0]].tolist()

# Wczytanie modelu
model = load_model('model_klas_words.h5')

# Wczytanie zapisanego tokenizera
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Funkcje z word_infinitive.py
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    lemmatizer = WordNetLemmatizer()
    words = word_tokenize(text)
    lemmatized_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words]
    return ' '.join(lemmatized_words)

def count_words(text):
    # Usuwanie znaków interpunkcyjnych i zmiana na małe litery
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    # Tokenizacja tekstu przy użyciu nltk
    words = word_tokenize(text)

    # Lematyzacja słów
    lemmatizer = WordNetLemmatizer()
    base_words = [lemmatizer.lemmatize(word, get_wordnet_pos(word)) for word in words if word.isalpha()]

    # Liczenie wystąpień słów
    word_counts = Counter(base_words)

    return word_counts

# Funkcja do wyznaczania słów kluczowych
def find_keywords(word_counts, stop_words, n=10):
    # Znalezienie n najczęstszych słów, pomijając stop words
    keywords = [word for word, count in word_counts.most_common() if word not in stop_words and len(word) > 2][:n]
    return keywords

# Funkcja do odczytu tekstu z pliku
def read_text_from_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

# Kod do klasyfikacji tekstu i wyodrębniania słów kluczowych
def classify_text_and_extract_keywords(text, model, tokenizer, number_key_words):
    # Lematyzacja i przygotowanie tekstu
    lemmatized_text = lemmatize_text(text)
    sequence = tokenizer.texts_to_sequences([lemmatized_text])
    padded_sequence = pad_sequences(sequence, maxlen=200)  # Ustaw maxlen zgodnie z modelem

    # Klasyfikacja tekstu
    prediction = model.predict(padded_sequence)
    predicted_class = prediction.argmax(axis=-1)

    predicted_class = predicted_class + 1

    # Wyodrębnienie słów kluczowych
    word_counts = count_words(lemmatized_text)
    keywords = find_keywords(word_counts, STOP_WORDS, number_key_words)

    return predicted_class, keywords, word_counts

# Ścieżka do pliku tekstowego 

file_choice = input("Wybierz, czy chcesz analizować tekst z pliku .txt(1) czy artykuł z danych testowych?(2): ").lower()
if file_choice == '1':
    csv_file = "text_english.txt"
    text_to_classify = read_text_from_file(csv_file)
else:
    random_row_index = random.randint(0, len(test_texts) - 1)
    random_row_index = 809
    text_to_classify = test_texts[random_row_index]
    label_to_classify = test_labels[random_row_index]

stop_words_file = "stop_words_english.txt"

# Pobieranie listy stop words z pliku
STOP_WORDS = read_text_from_file(stop_words_file)

number_words = 10 # Ilość najczęstszych słów w artykule wyświetlanych jako wynik
number_key_words = 5 # Ilość słów kluczowych wyświetlanych w artykule wyświetlanych jako wynik

predicted_class, keywords, word_counts = classify_text_and_extract_keywords(text_to_classify, model, tokenizer, number_key_words)

# Wyświetlenie wyników
print("\nNajczęstsze słowa:")
for word, count in word_counts.most_common(number_words):
        print(f"{word}: {count} wystąpień")

print(f"\nPredicted Class: {predicted_class}, Keywords: {keywords}")



