import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import os

MODEL_PATH = 'models/emotion_recognition_model.h5'
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "neutral", "sadness", "surprise"]

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Модель не найдена: {MODEL_PATH}")

emotion_model = load_model(MODEL_PATH)

def predict_image_emotion(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return [{"error": "Невозможно загрузить изображение"}]

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    results = []
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        face = face.astype("float") / 255.0
        face = img_to_array(face)
        face = np.expand_dims(face, axis=0)

        preds = emotion_model.predict(face)[0]
        emotion = EMOTIONS[preds.argmax()]
        confidence = float(preds.max())

        results.append({
            "emotion": emotion,
            "confidence": confidence,
            "bbox": [int(x), int(y), int(w), int(h)]
        })

    if not results:
        return [{"message": "Лицо не найдено"}]

    return results