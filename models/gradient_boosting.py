import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# Загрузка данных
data = pd.read_csv('datasets/custom_dataset.txt')
X = data['text']
y = data['class']

# Кодирование меток
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Векторизация текста
vectorizer = TfidfVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, y_encoded, test_size=0.2, random_state=42)

# Обучение модели
model = GradientBoostingClassifier()
model.fit(X_train, y_train)

# Расчет accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Предсказание текста
def predict_text(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return le.inverse_transform(prediction)[0]