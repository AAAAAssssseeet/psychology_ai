import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter

# Загрузка данных
data = pd.read_csv('datasets/test.txt', sep=';', names=["text", "emotion"])

# Кодирование меток
le = LabelEncoder()
y = le.fit_transform(data['emotion'])

# Векторизация текста
vectorizer = CountVectorizer(binary=True)  # Используем бинарные признаки: есть слово (1) или нет (0)
X_vectorized = vectorizer.fit_transform(data['text'])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_vectorized.toarray(), y, test_size=0.2, random_state=42)

# Реализация дерева решений
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # если лист - запоминаем предсказание

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, X, y):
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(set(y))

        # Стоп условия
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        # Ищем лучшее разбиение
        best_feature, best_threshold = self._best_split(X, y, n_features)

        # Строим дерево
        left_idxs = X[:, best_feature] == 0
        right_idxs = X[:, best_feature] == 1
        left = self._grow_tree(X[left_idxs], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs], y[right_idxs], depth + 1)
        return Node(feature=best_feature, threshold=best_threshold, left=left, right=right)

    def _best_split(self, X, y, n_features):
        best_gini = 1.0
        best_feature = None
        for feature_idx in range(n_features):
            thresholds = [0, 1]
            for threshold in thresholds:
                left_idxs = X[:, feature_idx] <= threshold
                right_idxs = X[:, feature_idx] > threshold

                if len(y[left_idxs]) == 0 or len(y[right_idxs]) == 0:
                    continue

                gini = self._gini_index(y[left_idxs], y[right_idxs])

                if gini < best_gini:
                    best_gini = gini
                    best_feature = feature_idx

        return best_feature, 0.5  # Порог всегда 0.5 для бинарных признаков (0/1)

    def _gini_index(self, left_labels, right_labels):
        def gini(labels):
            counts = np.bincount(labels)
            probs = counts / len(labels)
            return 1 - np.sum(probs ** 2)

        n = len(left_labels) + len(right_labels)
        gini_left = gini(left_labels)
        gini_right = gini(right_labels)
        return (len(left_labels) / n) * gini_left + (len(right_labels) / n) * gini_right

    def _most_common_label(self, y):
        counter = Counter(y)
        return counter.most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] == 0:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)

# Обучение модели
tree = DecisionTree(max_depth=10)
tree.fit(X_train, y_train)

# Оценка модели
y_pred = tree.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.2f}")

# Предсказание текста
def predict_text(text):
    text_vectorized = vectorizer.transform([text]).toarray()
    pred = tree.predict(text_vectorized)
    return le.inverse_transform(pred)[0]

# Пример использования
example_text = "I am very happy today!"
print(f"Predicted emotion: {predict_text(example_text)}")
