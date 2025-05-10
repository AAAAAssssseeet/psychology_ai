import numpy as np
import re


# 1. PCA для уменьшения размерности данных
def pca_dimensionality_reduction(data, n_components=2):
    """
    Уменьшает размерность данных с помощью PCA.

    Параметры:
        data: numpy array, исходные данные (формат: [n_samples, n_features])
        n_components: int, количество компонент для вывода (по умолчанию 2)

    Возвращает:
        transformed_data: numpy array, данные в новом пространстве признаков
    """
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    covariance_matrix = np.cov(centered_data, rowvar=False)
    eigen_values, eigen_vectors = np.linalg.eig(covariance_matrix)
    sorted_indices = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vectors[:, sorted_indices]
    projection_matrix = sorted_eigen_vectors[:, :n_components]
    transformed_data = np.dot(centered_data, projection_matrix)
    return transformed_data


# 2. Простой анализ эмоций по ключевым словам
def predict_text(text):
    """
    Анализирует текст и возвращает предполагаемые эмоции на основе ключевых слов

    Параметры:
        text (str): Входной текст для анализа

    Возвращает:
        list: Список словарей с эмоциями и уровнем уверенности
    """
    # База ключевых слов для определения эмоций
    EMOTION_KEYWORDS = {
        "joy": ["happy", "cheerful", "wonderful", "great", "hooray"],
        "sadness": ["sad", "bad", "melancholy", "bored", "lonely"],
        "anger": ["angry", "irritated", "annoyed", "hate", "furious"],
        "surprise": ["unexpected", "strange", "amazing", "wow", "can't believe it"],
        "fear": ["afraid", "scary", "horror", "anxiety", "panic"]
    }

    # Очистка текста
    cleaned = clean_text(text)
    if not cleaned:
        return [{"error": "Пустой текст после очистки"}]

    # Подсчет совпадений ключевых слов
    emotion_scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    words = cleaned.split()

    for word in words:
        for emotion, keywords in EMOTION_KEYWORDS.items():
            if word in keywords:
                emotion_scores[emotion] += 1

    # Расчет уверенности
    total_words = len(words)
    results = []
    for emotion, count in emotion_scores.items():
        confidence = round(count / total_words * 100 if total_words > 0 else 0, 2)
        if confidence > 0:
            results.append({
                "emotion": emotion,
                "confidence": confidence
            })

    # Сортировка по уверенности
    return sorted(results, key=lambda x: x['confidence'], reverse=True) or [
        {"emotion": "нейтральный", "confidence": 100}]


def clean_text(text):
    """
    Очищает текст от специальных символов и нормализует

    Параметры:
        text (str): Исходный текст

    Возвращает:
        str: Очищенный текст
    """
    if not isinstance(text, str):
        return ""

    text = text.strip().lower()
    text = re.sub(r'[^\w\s]', '', text)  # Удаление пунктуации
    text = re.sub(r'\d+', '', text)  # Удаление чисел
    return ' '.join(text.split())