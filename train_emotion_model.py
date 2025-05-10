import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Параметры
IMG_SIZE = 48
BATCH_SIZE = 32
EPOCHS = 10
EMOTIONS = ["anger", "contempt", "disgust", "fear", "happy", "sadness", "surprise"]

# Генератор данных
datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)

train_gen = datagen.flow_from_directory(
    'datasets/CK +48',  # Путь к папке с изображениями
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='training'  # Указание на выборку для обучения
)

val_gen = datagen.flow_from_directory(
    'datasets/CK +48',  # Путь к папке с изображениями
    target_size=(IMG_SIZE, IMG_SIZE),
    color_mode='grayscale',
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    subset='validation'  # Указание на выборку для валидации
)


# Функция извлечения признаков с помощью SIFT
def extract_sift_features(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    sift = cv2.SIFT_create()
    keypoints, descriptors = sift.detectAndCompute(img, None)
    return descriptors


# Функция для загрузки изображения и извлечения признаков SIFT
def sift_feature_pipeline(image_path, img_size=(48, 48)):
    img = load_img(image_path, target_size=img_size, color_mode='grayscale')
    img_array = img_to_array(img)
    descriptors = extract_sift_features(image_path)

    if descriptors is None:
        descriptors = np.zeros((1, 128))  # Если нет признаков, заполняем нулями

    image_flattened = img_array.flatten()
    combined_features = np.concatenate((image_flattened, descriptors.flatten()))

    return combined_features


# Подготовка данных для обучения (извлечение признаков SIFT и изображений)
train_images = []
train_sift_features = []
train_labels = []

# Перебираем все изображения из генератора и извлекаем признаки
for i, (image, label) in enumerate(train_gen):
    image_path = train_gen.filepaths[i]  # Получаем путь к изображению
    image_features = sift_feature_pipeline(image_path)
    train_images.append(image.flatten())  # Добавляем изображение в плоском виде
    train_sift_features.append(image_features)  # Добавляем признаки SIFT
    train_labels.append(label)  # Добавляем метки эмоций

# Преобразуем в массивы NumPy
train_images = np.array(train_images)
train_sift_features = np.array(train_sift_features)
train_labels = np.array(train_labels)

# Для валидационной выборки
val_images = []
val_sift_features = []
val_labels = []

for i, (image, label) in enumerate(val_gen):
    image_path = val_gen.filepaths[i]  # Получаем путь к изображению
    image_features = sift_feature_pipeline(image_path)
    val_images.append(image.flatten())
    val_sift_features.append(image_features)
    val_labels.append(label)

# Преобразуем в массивы NumPy для валидации
val_images = np.array(val_images)
val_sift_features = np.array(val_sift_features)
val_labels = np.array(val_labels)

# Модель
image_input = Input(shape=(IMG_SIZE, IMG_SIZE, 1))  # Вход для изображения
x = Conv2D(32, (3, 3), activation='relu')(image_input)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(64, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Conv2D(128, (3, 3), activation='relu')(x)
x = MaxPooling2D((2, 2))(x)
x = Flatten()(x)

# Вход для признаков SIFT
sift_input = Input(shape=(train_sift_features.shape[1],))  # Размер признаков SIFT
x = Concatenate()([x, sift_input])  # Объединяем признаки SIFT и изображение
x = Dense(128, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(len(EMOTIONS), activation='softmax')(x)

# Компиляция модели
model = Model(inputs=[image_input, sift_input], outputs=x)
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit([train_images, train_sift_features], train_labels,
          validation_data=([val_images, val_sift_features], val_labels), epochs=EPOCHS)

# Сохранение модели
model.save('models/emotion_recognition_model_with_sift.h5')
