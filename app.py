from flask import Flask, render_template, request, redirect, url_for
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Словарь: отображение названий моделей в интерфейсе на имена соответствующих модулей
models = {
    "Linear Regression": "linear_regression",
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "Random Forest": "random_forest",
    "Naive Bayes": "naive_bayes",
    "K-Nearest Neighbors": "knn",
    "Support Vector Machine (SVM)": "svm",
    "Gradient Boosting": "gradient_boosting",
    "KMeans Clustering": "kmeans_clustering",
    "Principal Component Analysis (PCA)": "pca_dimensionality_reduction",
    "Apriori Association": "apriori_association",
    "Image Emotion Recognition": "image_emotion_recognition"
}

@app.route('/')
def index():
    return render_template('index.html', models=models)

@app.route('/predict/<model_name>', methods=['GET', 'POST'])
def predict(model_name):
    if model_name not in models:
        return "Модель не найдена", 404

    module_name = models[model_name]

    if request.method == 'POST':
        if module_name == "image_emotion_recognition":
            # Обработка изображений
            if 'image' not in request.files:
                return "Нет файла изображения"
            file = request.files['image']
            if file.filename == '':
                return "Файл не выбран"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)

            try:
                module = __import__(f"models.{module_name}", fromlist=['predict_image_emotion'])
            except ImportError:
                return f"Модуль {module_name} не найден", 500

            predictions = module.predict_image_emotion(filepath)
            results = [f"{p.get('emotion', '')} ({p.get('confidence', 0):.2f})" for p in predictions]
            return render_template('result.html',
                                 prediction=" | ".join(results),
                                 model_name=model_name,
                                 image_url=f"/{filepath}")
        else:
            # Обработка текста
            text = request.form['text']
            try:
                module = __import__(f"models.{module_name}", fromlist=['predict_text'])
            except ImportError:
                return f"Модель {module_name} не найдена", 500
            prediction = module.predict_text(text)
        return render_template('result.html', prediction=prediction, model_name=model_name)

    return render_template('predict.html', model_name=model_name)




if __name__ == '__main__':
    app.run(debug=True)