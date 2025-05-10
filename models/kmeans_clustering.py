import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import silhouette_score

# Загрузка данных
data = pd.read_csv('datasets/test.txt', sep=';', names=["text", "emotion"])
X = data['text']
y_true = data['emotion']

# Очистка данных
data = data.dropna()
data['emotion'] = data['emotion'].str.strip().str.lower()
valid_emotions = ['happy', 'sad', 'fear', 'anger', 'neutral']  # Допустимые эмоции
data = data[data['emotion'].isin(valid_emotions)]  # Фильтруем только нужные эмоции

# Векторизация текста
vectorizer = CountVectorizer(max_features=500, stop_words='english')
X_vectorized = vectorizer.fit_transform(X)

# Обучение модели
n_clusters = len(valid_emotions)  # Количество кластеров = количествоу эмоций
model = KMeans(n_clusters=n_clusters, random_state=42)
model.fit(X_vectorized)

# Оценка качества
silhouette_avg = silhouette_score(X_vectorized, model.labels_)
print(f"Silhouette Score: {silhouette_avg:.2f}")

# Сопоставление кластеров с эмоциями
cluster_to_emotion = {}
for cluster in range(n_clusters):
    mask = model.labels_ == cluster
    cluster_emotions = y_true[mask]
    if not cluster_emotions.empty:
        cluster_to_emotion[cluster] = cluster_emotions.mode()[0]
    else:
        cluster_to_emotion[cluster] = "unknown"


# Предсказание с несколькими вариантами
def predict_text(text):
    text_vectorized = vectorizer.transform([text])

    # Получаем расстояние до каждого кластера
    distances = model.transform(text_vectorized)

    # Сортируем кластеры по расстоянию
    cluster_distances = list(enumerate(distances[0]))
    cluster_distances.sort(key=lambda x: x[1])  # По возрастанию расстояния

    # Рассчитываем уверенность
    results = []
    for cluster_id, distance in cluster_distances[:3]:  # Топ-3 ближайших кластера
        emotion = cluster_to_emotion.get(cluster_id, "unknown")
        confidence = 1 / (1 + distance)  # Нормализация расстояния в уверенность
        results.append({
            "cluster": cluster_id,
            "emotion": emotion,
            "distance": round(distance, 2),
            "confidence": round(confidence, 2)
        })

    # Сортировка по уверенности
    results.sort(key=lambda x: x['confidence'], reverse=True)

    # Формируем вывод
    output = "Вероятные эмоции:\n"
    for res in results:
        output += f"Кластер {res['cluster']} → {res['emotion'].capitalize()} | Уверенность: {res['confidence'] * 100:.1f}%\n"

    return output