# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичем
# Файл реализации линейной рециркуляционной сети
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач. Практикум: учебно-методическое пособие / В.П. Ивашенко. – Минск: БГУИР, 2020.

import numpy as np
from PIL import Image
from model import ImageReconstructorNN
from data_utils import pad_and_chunk_image, reconstruct_image_from_chunks

INPUT_IMAGE_PATH = 'input.bmp'
OUTPUT_IMAGE_PATH = 'output.bmp'

CHUNK_SHAPE = (8, 8)  # Размер входного окна (чанка)
# Входной слой будет равен 8*8*3 = 192 нейрона.
HIDDEN_SIZE = 64

TRAINING_CONFIG = {
    # 1. Цель обучения: остановиться, когда средняя ошибка (MSE) станет меньше 0.005.
    'target_loss': 0.1,
    # 2. Предохранитель: если цель не достигнута, остановиться через 50000 эпох.
    'max_epochs': 50000,
    # 3. Скорость обучения (начальная).
    'initial_lr': 0.05,
    # 4. Как часто логировать в консоль.
    'log_frequency': 100,
    # 5. Настройки планировщика коэффициента обучения.
    'enable_scheduler': True,
    'lr_scheduler_patience_epochs': 100,  # Ждать 500 эпох без рекордов перед снижением LR
    'lr_factor': 0.5,  # Уменьшать LR в 2 раза
    'min_lr': 1e-6  # Не опускаться ниже этого значения
}

if __name__ == '__main__':
    try:
        # 1. Загрузка и нормализация изображения (0..255 -> 0..1)
        original_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
        image_array = np.array(original_image) / 255.0

        # 2. Нарезка изображения на обучающие примеры (чанки) и дополнение краев
        chunk_sequence, padded_shape, original_shape = pad_and_chunk_image(image_array, CHUNK_SHAPE)

        # Входной слой определяется автоматически данными (192)
        input_size = chunk_sequence.shape[1]

        # 3. Создание модели
        model = ImageReconstructorNN(input_size, HIDDEN_SIZE, input_size)

        print(f"Image original size: {original_shape[1]}x{original_shape[0]}")
        print(f"Vector size per chunk (Input): {input_size}")
        print(f"Hidden layer size (Bottleneck): {HIDDEN_SIZE}")

        # 4. Запуск обучения (передаем данные и конфиг)
        # В автоэнкодере вход (chunk_sequence) и цель (chunk_sequence) совпадают.
        model.train(chunk_sequence, chunk_sequence, TRAINING_CONFIG)

        print("Training finished. Reconstructing image...")

        # 5. Использование обученной модели (Inference)
        # Важно привести вход к float32, так как веса модель в float32
        reconstructed_chunks_flat = model.forward(chunk_sequence.astype(np.float32))
        reconstructed_chunks = np.squeeze(np.array(reconstructed_chunks_flat))

        # 6. Сборка изображения из кусочков
        reconstructed_padded_array = reconstruct_image_from_chunks(reconstructed_chunks, padded_shape, CHUNK_SHAPE)

        # 7. Обрезка лишних полей (padding), которые добавлялись в начале
        h_orig, w_orig, _ = original_shape
        reconstructed_final_array = reconstructed_padded_array[:h_orig, :w_orig, :]

        # 8. Сохранение результата
        # Ограничиваем значения 0..1 и переводим в байты 0..255
        reconstructed_final_array = np.clip(reconstructed_final_array, 0, 1)
        reconstructed_final_array = (reconstructed_final_array * 255).astype(np.uint8)
        result_image = Image.fromarray(reconstructed_final_array, 'RGB')

        result_image.save(OUTPUT_IMAGE_PATH)
        print(f"Reconstructed image saved to: {OUTPUT_IMAGE_PATH}")
        result_image.show()

    except FileNotFoundError:
        print(f"Error: File not found at '{INPUT_IMAGE_PATH}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
