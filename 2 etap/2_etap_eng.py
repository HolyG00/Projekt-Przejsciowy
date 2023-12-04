import pandas as pd
import string
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')  # Pobierz tokenizator Punkt, jeśli nie jest jeszcze zainstalowany
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

# Funkcja do odczytu tekstu z pliku
def read_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Funkcja do wczytywania danych z pliku CSV
def read_data_from_csv(filename):
    df = pd.read_csv(filename, header=None, names=['Label', 'Title', 'Text'])
    return df

# Funkcja do przetwarzania tekstu i zliczania słów
def count_words(text):
    # Usuwanie znaków interpunkcyjnych i zmiana na małe litery
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    # Tokenizacja tekstu przy użyciu nltk i usunięcie dodatkowych białych znaków
    words = [word.strip() for word in nltk.word_tokenize(text)]

    # Używanie generatora zamiast tworzenia listy
    base_words = (word.rstrip('s') if word.endswith('s') else word for word in words)

    # Lematyzacja słów
    lemmatizer = WordNetLemmatizer()
    base_words = [lemmatizer.lemmatize(word) for word in base_words]

    # Liczenie wystąpień słów
    word_counts = Counter(base_words)

    return word_counts

# Funkcja do wyznaczania słów kluczowych
def find_keywords(word_counts, stop_words, n=5):
    # Znalezienie n najczęstszych słów, pomijając stop words
    keywords = [word for word, count in word_counts.most_common() if word not in stop_words and len(word) >= 2][:n]
    return keywords

csv_file = "train.csv"
stop_words_file = "stop_words_english.txt"

# Pobieranie listy stop words z pliku
STOP_WORDS = read_text_from_file(stop_words_file)

# Wczytanie danych z pliku CSV
df = read_data_from_csv(csv_file)

# Wczytanie danych z pliku CSV
df = read_data_from_csv(csv_file)

# Wybór losowego wiersza do analizy
random_row = df.sample(1)

# Zliczanie słów
word_counts = count_words(random_row['Text'].iloc[0])

# Wyznaczanie słów kluczowych (5 najczęstszych)
keywords = find_keywords(word_counts, STOP_WORDS, 5)

# Wyświetlenie wyników
print("\nLosowy wiersz:")
print(random_row[['Title', 'Text']])
print("\nNajczęstsze słowa:")
for word, count in word_counts.most_common(10):
    if len(word) >= 2:
        print(f"{word}: {count} wystąpień")

print("\nSłowa kluczowe:")
print(keywords)
