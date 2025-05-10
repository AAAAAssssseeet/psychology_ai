import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
data = pd.read_csv('datasets/test.txt', sep=';', names=["text", "emotion"])

# Кодирование меток
le = LabelEncoder()
y = le.fit_transform(data['emotion'])

# Векторизация текста
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(data['text'])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y, test_size=0.2, random_state=42)

# Переводим X в плотный формат (матрица numpy), чтобы работать вручную
X_train = X_train.toarray()
X_test = X_test.toarray()

# Ручная реализация линейной регрессии (обучение)
# Решаем задачу: W = (X^T X)^-1 X^T y
X_train_aug = np.hstack((X_train, np.ones((X_train.shape[0], 1))))  # добавляем столбец единиц для смещения
W = np.linalg.pinv(X_train_aug.T @ X_train_aug) @ X_train_aug.T @ y_train  # псевдообратная матрица

# Предсказания на тесте
X_test_aug = np.hstack((X_test, np.ones((X_test.shape[0], 1))))  # добавляем столбец единиц для смещения
y_pred_continuous = X_test_aug @ W

# Округляем предсказания
y_pred = np.round(y_pred_continuous).astype(int)

# Оцениваем качество
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Предсказание текста
def predict_text(text):
    text_vectorized = vectorizer.transform([text]).toarray()
    text_aug = np.hstack((text_vectorized, np.ones((text_vectorized.shape[0], 1))))  # добавляем смещение
    prediction = text_aug @ W
    predicted_code = int(round(prediction[0]))
    return le.inverse_transform([predicted_code])[0]

# Пример предсказания
example_text = "I am very happy today"
print(f"Predicted emotion: {predict_text(example_text)}")
