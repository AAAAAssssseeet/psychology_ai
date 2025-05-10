import joblib
import re

# Загрузка обученных объектов
model = joblib.load('models/suicide_model.pkl')
vectorizer = joblib.load('models/vectorizer.pkl')
label_encoder = joblib.load('models/label_encoder.pkl')

def predict_text(text):
    """
    Предсказывает, является ли текст суицидальным или нет
    """
    if not text.strip():
        return {"error": "Пустой текст"}

    # Очистка текста
    text = re.sub(r'[^\w\s.,!?"\'@-]', '', text.strip().lower())

    try:
        # Векторизация
        X = vectorizer.transform([text])
        # Предсказание
        prediction = model.predict(X)[0]
        confidence = model.predict_proba(X).max()

        return {
            "class": label_encoder.inverse_transform([prediction])[0],
            "confidence": round(float(confidence), 2)
        }
    except Exception as e:
        return {"error": str(e)}