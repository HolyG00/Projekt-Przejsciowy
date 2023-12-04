import string
from collections import Counter

# Funkcja do odczytu tekstu z pliku
def read_text_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

# Funkcja do przetwarzania tekstu i zliczania słów
def count_words(text):
    # Usuwanie znaków interpunkcyjnych i zmiana na małe litery
    translator = str.maketrans('', '', string.punctuation)
    text = text.translate(translator).lower()

    # Podział tekstu na słowa
    words = text.split()

    # Liczenie wystąpień słów
    word_counts = Counter(words)

    return word_counts

# Funkcja do wyznaczania słów kluczowych
def find_keywords(word_counts, n=10):
    # Znalezienie n najczęstszych słów
    keywords = [word for word, count in word_counts.most_common(n) if (word not in STOP_WORDS and len(word) >= 2)]
    return keywords

# Lista spójników do wykluczenia
STOP_WORDS = ["a", "i", "lub", "oraz", "ale", "czy", "gdy", "jeśli", "bo", "przy","jest", "ponieważ"]

# Odczyt tekstu z pliku
text = read_text_from_file("text_pl.txt")

# Zliczanie słów
word_counts = count_words(text)

# Wyznaczanie słów kluczowych (10 najczęstszych)
keywords = find_keywords(word_counts, 10)

# Wyświetlenie wyników
print("Najczęstsze słowa:")
for word, count in word_counts.most_common(10):
    print(f"{word}: {count} wystąpień")

print("\nSłowa kluczowe:")
print(keywords)
