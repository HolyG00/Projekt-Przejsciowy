import string
from collections import Counter
import nltk
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

# Funkcja do przetwarzania tekstu i zliczania słów
def count_words(text, language):
    # Usuwanie znaków interpunkcyjnych i zmiana na małe litery
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    # Tokenizacja tekstu przy użyciu nltk i usunięcie dodatkowych białych znaków
    words = [word.strip() for word in nltk.word_tokenize(text)]

    # Używanie generatora zamiast tworzenia listy
    if language == 'polski':
        base_words = (word.rstrip('eiouąęyńć') if word.endswith(('u', 'i', 'owi', 'emu', 'em', 'y', 'ego', 'ą', 'ę', 'ń', 'ć')) else word for word in words)
    else:
        base_words = (word.rstrip('s') if word.endswith('s') else word for word in words)

    base_words = words

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

# Wybór języka przez użytkownika
language = input("Wybierz język (polski/angielski): ").lower()

# Wybór pliku stop words na podstawie wybranego języka
if language == 'polski':
    stop_words_file = "stop_words_polski.txt"
    text_file = "text_polski.txt"

elif language == 'angielski':
    stop_words_file = "stop_words_english.txt"
    text_file = "text_english.txt"
else:
    print("Niepoprawny wybór języka. Program zostanie zakończony.")
    exit()

# Pobieranie listy stop words z pliku
STOP_WORDS = read_text_from_file(stop_words_file)

# Odczyt tekstu z pliku
text = read_text_from_file(text_file)

# Zliczanie słów
word_counts = count_words(text, language)

# Wyznaczanie słów kluczowych (5 najczęstszych)
keywords = find_keywords(word_counts, STOP_WORDS, 5)

# Wyświetlenie wyników
print("\nNajczęstsze słowa:")
for word, count in word_counts.most_common(10):
    if len(word) >= 2:
        print(f"{word}: {count} wystąpień")

print("\nSłowa kluczowe:")
print(keywords)

# Zapis do pliku wynikowego
output_file_name = f"wynik_{language}.txt"
with open(output_file_name, 'w', encoding='utf-8') as output_file:
    output_file.write("Najczęstsze słowa:\n")
    for word, count in word_counts.most_common(5):
        if len(word) >= 2:
            output_file.write(f"{word}: {count} wystąpień\n")

    output_file.write("\nSłowa kluczowe:\n")
    for keyword in keywords:
        output_file.write(f"{keyword}\n")

print(f"Wyniki zostały zapisane do pliku: {output_file_name}")
