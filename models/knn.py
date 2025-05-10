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

# Свой KNN-классификатор
class SimpleKNNClassifier:
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = []
        for x in X:
            distances = np.linalg.norm(self.X_train - x, axis=1)  # Евклидово расстояние
            neighbor_indices = np.argsort(distances)[:self.n_neighbors]
            neighbor_labels = self.y_train[neighbor_indices]
            # Выбираем наиболее частую метку среди соседей
            unique, counts = np.unique(neighbor_labels, return_counts=True)
            most_common_label = unique[np.argmax(counts)]
            predictions.append(most_common_label)
        return np.array(predictions)

# Функция разбиения данных
def train_test_split_manual(X, y, test_size=0.2, random_state=None):
    np.random.seed(random_state)
    indices = np.random.permutation(len(X))
    test_size = int(len(X) * test_size)
    test_idx, train_idx = indices[:test_size], indices[test_size:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]

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
X_train, X_test, y_train, y_test = train_test_split_manual(X_vectorized, y, test_size=0.2, random_state=42)

# Обучение модели
model = SimpleKNNClassifier(n_neighbors=5)
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
example = "I feel very happy today"
print(predict_text(example))
