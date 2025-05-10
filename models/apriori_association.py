# models/apriori_association.py

import pandas as pd
from mlxtend.frequent_patterns import apriori
from sklearn.feature_extraction.text import CountVectorizer

# 1. Загрузка данных
data = pd.read_csv('datasets/test.txt', sep=';', names=["text", "emotion"])

# 2. Создание бинарной матрицы (эмодзи как товары)
df = pd.get_dummies(data['emotion'])

# 3. Поиск частых ассоциаций (как раньше)
frequent_itemsets = apriori(df, min_support=0.1, use_colnames=True)

# 4. Простая система предсказания эмоций по тексту
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['text'])
y = data['emotion']

# Словарь: слова -> эмоции
word_to_emotion = {}
for _, row in data.iterrows():
    words = row['text'].lower().split()
    for word in words:
        if word not in word_to_emotion:
            word_to_emotion[word] = {}
        word_to_emotion[word][row['emotion']] = word_to_emotion[word].get(row['emotion'], 0) + 1

def predict_text(text):
    """
    Предсказывает эмоцию по тексту или возвращает ассоциации между эмоциями
    """
    cleaned = clean_text(text)
    if not cleaned:
        return [{"message": "Пустой текст"}]

    # Если текст содержит слова — пытаемся предсказать эмоцию
    words = cleaned.split()
    emotion_scores = {}

    for word in words:
        if word in word_to_emotion:
            for emotion, count in word_to_emotion[word].items():
                emotion_scores[emotion] = emotion_scores.get(emotion, 0) + count

    if emotion_scores:
        total = sum(emotion_scores.values())
        result = [
            {"emotion": e, "confidence": round(c / total * 100, 2)}
            for e, c in sorted(emotion_scores.items(), key=lambda x: x[1], reverse=True)
        ]
        return result

    # Если нет совпадений — возвращаем ассоциации между эмоциями
    return [{
        "message": "Эта модель показывает ассоциации между эмоциями, а не предсказывает их.",
        "hint": "Попробуйте ввести текст с ключевыми словами (например: грустно, весело)"
    }]

def clean_text(text):
    """Очистка текста"""
    if not isinstance(text, str):
        return ""
    text = text.strip().lower()
    text = ''.join([c for c in text if c.isalnum() or c.isspace()])
    return ' '.join(text.split())