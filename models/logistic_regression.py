import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
data = pd.read_csv('datasets/test.txt', sep=';', names=["text", "emotion"])

# Кодирование меток
le = LabelEncoder()
y = le.fit_transform(data['emotion'])

# Векторизация текста (TF-IDF вместо Count)
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(data['text'])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Переводим X в numpy
X_train = X_train.toarray()
X_test = X_test.toarray()

# Параметры
n_classes = len(np.unique(y_train))
n_features = X_train.shape[1]
learning_rate = 0.01
n_epochs = 3000
reg_strength = 0.001  # регуляризация

# Инициализация весов
W = np.zeros((n_classes, n_features + 1))

# Добавляем столбец единиц
X_train_aug = np.hstack((X_train, np.ones((X_train.shape[0], 1))))
X_test_aug = np.hstack((X_test, np.ones((X_test.shape[0], 1))))

# One-hot кодирование
y_train_one_hot = np.zeros((y_train.size, n_classes))
y_train_one_hot[np.arange(y_train.size), y_train] = 1

# Сигмоида
def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Обучение
for epoch in range(n_epochs):
    logits = X_train_aug @ W.T
    probs = softmax(logits)
    error = probs - y_train_one_hot
    grad = (error.T @ X_train_aug) / X_train.shape[0] + reg_strength * W
    W -= learning_rate * grad

# Предсказание
def predict(X):
    logits = X @ W.T
    probs = softmax(logits)
    return np.argmax(probs, axis=1), np.max(probs, axis=1)  # возвращаем и класс и уверенность

# Оценка
y_pred, _ = predict(X_test_aug)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Предсказание текста с уверенностью
def predict_text(text):
    text_vectorized = vectorizer.transform([text]).toarray()
    text_aug = np.hstack((text_vectorized, np.ones((text_vectorized.shape[0], 1))))
    pred_class, confidence = predict(text_aug)
    emotion = le.inverse_transform(pred_class)[0]
    return emotion, confidence[0]

# Пример
example_text = "I feel very angry today"
emotion, confidence = predict_text(example_text)
print(f"Predicted emotion: {emotion} ({confidence * 100:.2f}% confident)")
