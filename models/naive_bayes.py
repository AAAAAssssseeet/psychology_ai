import pandas as pd
import numpy as np
import re

# Простейший векторизатор
class SimpleVectorizer:
    def __init__(self):
        self.vocabulary_ = {}

    def fit(self, texts):
        words = set()
        for text in texts:
            text = self._preprocess(text)
            words.update(text.split())
        self.vocabulary_ = {word: idx for idx, word in enumerate(sorted(words))}

    def transform(self, texts):
        X = np.zeros((len(texts), len(self.vocabulary_)))
        for i, text in enumerate(texts):
            text = self._preprocess(text)
            for word in text.split():
                if word in self.vocabulary_:
                    X[i, self.vocabulary_[word]] += 1
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

# Свой Multinomial Naive Bayes
class SimpleMultinomialNB:
    def __init__(self):
        self.class_log_prior_ = None
        self.feature_log_prob_ = None
        self.classes_ = None

    def fit(self, X, y):
        count_sample = X.shape[0]
        self.classes_ = np.unique(y)
        class_count = np.array([np.sum(y == c) for c in self.classes_])
        self.class_log_prior_ = np.log(class_count) - np.log(count_sample)

        # Считаем количество слов в каждом классе
        feature_count = np.zeros((len(self.classes_), X.shape[1]))
        for idx, c in enumerate(self.classes_):
            feature_count[idx, :] = np.sum(X[y == c], axis=0)

        # Лапласовская коррекция (плюс 1)
        smoothed_fc = feature_count + 1
        smoothed_cc = smoothed_fc.sum(axis=1).reshape(-1, 1)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)

    def predict(self, X):
        jll = self._joint_log_likelihood(X)
        return self.classes_[np.argmax(jll, axis=1)]

    def _joint_log_likelihood(self, X):
        return X @ self.feature_log_prob_.T + self.class_log_prior_

# Загрузка данных
data = pd.read_csv('datasets/test.txt', sep=';', names=["text", "emotion"])

# Кодирование меток
le = SimpleLabelEncoder()
y = le.fit(data['emotion'])
y = le.transform(data['emotion'])

# Векторизация текста
vectorizer = SimpleVectorizer()
X_vectorized = vectorizer.fit_transform(data['text'])

# Разделение данных
def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

X_train, X_test, y_train, y_test = train_test_split_manual(X_vectorized, y, test_size=0.2, random_state=42)

# Обучение модели
model = SimpleMultinomialNB()
model.fit(X_train, y_train)

# Расчет accuracy
y_pred = model.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Предсказание текста
def predict_text(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return le.inverse_transform(prediction)[0]

# Пример
example = "I feel very sad today"
print(predict_text(example))
