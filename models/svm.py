import pandas as pd
import numpy as np
import re

# Простейший TF-IDF векторизатор
class SimpleTfidfVectorizer:
    def __init__(self):
        self.vocabulary_ = {}
        self.idf_ = {}

    def fit(self, texts):
        df = {}
        corpus_size = len(texts)
        for text in texts:
            text = self._preprocess(text)
            words = set(text.split())
            for word in words:
                df[word] = df.get(word, 0) + 1
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(df.keys()))}
        self.idf_ = {word: np.log((corpus_size + 1) / (freq + 1)) + 1 for word, freq in df.items()}  # сглаживание

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocabulary_)))
        for i, text in enumerate(texts):
            text = self._preprocess(text)
            word_counts = {}
            words = text.split()
            for word in words:
                if word in self.vocabulary_:
                    word_counts[word] = word_counts.get(word, 0) + 1
            for word, count in word_counts.items():
                idx = self.vocabulary_[word]
                tf = count / len(words)
                X[i, idx] = tf * self.idf_.get(word, 0)
        return X

    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def _preprocess(self, text):
        return re.sub(r'[^\w\s]', '', text.strip().lower())

# Простейший кодировщик меток
class SimpleLabelEncoder:
    def __init__(self):
        self.classes_ = []

    def fit(self, labels):
        self.classes_ = sorted(set(labels))

    def transform(self, labels):
        return np.array([self.classes_.index(label) for label in labels])

    def inverse_transform(self, indices):
        return [self.classes_[i] for i in indices]

# Простейшая линейная SVM
class SimpleLinearSVM:
    def __init__(self, lr=0.01, epochs=1000, C=1.0):
        self.lr = lr
        self.epochs = epochs
        self.C = C
        self.weights = None
        self.bias = 0

    def fit(self, X, y):
        n_samples, n_features = X.shape
        y_ = np.where(y == 1, 1, -1)

        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    self.weights -= self.lr * (2 * self.C * self.weights)
                else:
                    self.weights -= self.lr * (2 * self.C * self.weights - np.dot(x_i, y_[idx]))
                    self.bias -= self.lr * y_[idx]

    def decision_function(self, X):
        return np.dot(X, self.weights) + self.bias

    def predict(self, X):
        decision = self.decision_function(X)
        return np.where(decision > 0, 1, 0)

    def predict_proba(self, X):
        decision = self.decision_function(X)
        proba = 1 / (1 + np.exp(-decision))
        return np.vstack([1 - proba, proba]).T

def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

# === ОСНОВНОЙ КОД ===

# Загрузка двух датасетов
data1 = pd.read_csv('datasets/custom_dataset.txt')
data2 = pd.read_csv('datasets/second_dataset.txt')

data = pd.concat([data1, data2], ignore_index=True)
X = data['text'].tolist()
y = data['class'].tolist()

# Перевод sadness, anger, fear в 'negative'
def relabel(label):
    if label in ['sadness', 'anger', 'fear']:
        return 'negative'
    return label

y = [relabel(label) for label in y]

# Кодирование меток
le = SimpleLabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

# Векторизация текста
vectorizer = SimpleTfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split_manual(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Обучение модели
model = SimpleLinearSVM(lr=0.001, epochs=1000, C=1.0)
model.fit(X_train, y_train)

# Оценка точности
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Предсказание текста
def predict_text(text):
    text_vectorized = vectorizer.transform([text])
    pred = model.predict(text_vectorized)[0]
    probas = model.predict_proba(text_vectorized)[0]
    label = le.inverse_transform([pred])[0]
    confidence = round(probas[pred] * 100, 2)
    return {"class": label, "confidence": f"{confidence}%"}

# Пример
example = "I feel very sad today"
print(predict_text(example))
