# Индивидуальная лабораторная работа 2 по дисциплине МРЗвИС вариант 6
# "Реализовать модель сети Хэмминга"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import os

import config
import numpy as np
from PIL import Image


def load_image_as_vector(filepath, target_size=None):
    """
    Загружает изображение.
    Если target_size задан (tuple (w, h)), делает ресайз к нему.
    Если target_size is None, использует оригинальный размер.

    Возвращает: (vector, (width, height))
    """
    try:
        img = Image.open(filepath).convert("L")

        # Ресайз под первый попавшийся эталон в датасете.
        if target_size is not None:
            img = img.resize(target_size)

        width, height = img.size
        arr = np.array(img)

        # Бинаризация: < Threshold -> 1 (инфо), >= Threshold -> -1 (фон)
        binary = np.where(arr < config.BINARY_THRESHOLD, 1, -1)

        return binary.flatten(), (width, height)
    except Exception as e:
        print(f"Ошибка чтения {filepath}: {e}")
        return None, None


def vector_to_matrix(vector, shape):
    """
    Превращает вектор обратно в матрицу для отрисовки.
    shape: кортеж (width, height)
    """
    w, h = shape
    # Reshape требует (height, width) для numpy
    matrix = vector.reshape((h, w))

    display_img = np.zeros_like(matrix)
    display_img[matrix == 1] = 0  # Черный
    display_img[matrix == -1] = 255  # Белый

    return display_img


def add_noise(vector, noise_level):
    """
    Инвертирует случайные биты в векторе.
    """
    noisy = np.copy(vector)
    n_pixels = len(vector)
    n_noise = int(n_pixels * noise_level)

    if n_noise > 0:
        # Выбираем случайные индексы без повторений и инвертируем (1 -> -1, -1 -> 1).
        indices = np.random.choice(n_pixels, n_noise, replace=False)
        noisy[indices] *= -1

    return noisy


def generate_dummy_data():
    """
    Генерирует тестовые данные, используя размер по умолчанию из конфига.
    """
    if not os.path.exists(config.DATA_DIR):
        os.makedirs(config.DATA_DIR)

    if len(os.listdir(config.DATA_DIR)) == 0:
        print(f"Генерация тестовых данных ({config.DEFAULT_GEN_SIZE})...")
        chars = ["A", "B", "C", "D", "E"]
        np.random.seed(42)
        w, h = config.DEFAULT_GEN_SIZE
        for char in chars:
            img_arr = np.random.choice([0, 255], size=(h, w)).astype(np.uint8)
            Image.fromarray(img_arr).save(os.path.join(config.DATA_DIR, f"{char}.png"))
