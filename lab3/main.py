# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичем
# Файл реализации линейной рециркуляционной сети
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач. Практикум: учебно-методическое пособие / В.П. Ивашенко. – Минск: БГУИР, 2020.

import numpy as np
from PIL import Image
from model import ImageReconstructorRNN
from data_utils import pad_and_chunk_image, reconstruct_image_from_chunks

INPUT_IMAGE_PATH = 'input.bmp'
OUTPUT_IMAGE_PATH = 'output.bmp'

CHUNK_SHAPE = (8, 8)  # Размер одного чанка (кусочка) изображения в пикселях.
HIDDEN_SIZE = 256  # Размер "памяти" нейросети (количество нейронов в скрытом слое).
EPOCHS = 500  # Общее количество раз, которое модель "посмотрит" на изображение для обучения.
LOG_FREQUENCY = 1  # Как часто выводить информацию о потерях (каждую 1 эпоху).

INITIAL_LEARNING_RATE = 0.001  # Начальный коэффициент обучения (скорость обучения).

# Настройки для планировщика коэффициента обучения (Learning Rate Scheduler).
ENABLE_LR_SCHEDULER = True  # Включить/выключить автоматическое снижение скорости обучения.
LR_PATIENCE_PERCENTAGE = 1.0  # Процент от EPOCHS, который нужно подождать без улучшений, прежде чем снижать LR.
LR_SCHEDULER_FACTOR = 0.5  # Множитель для снижения LR (0.5 означает уменьшение в 2 раза).
MIN_LEARNING_RATE = 1e-6  # Минимальное значение LR, ниже которого он опускаться не будет.

if __name__ == '__main__':
    try:
        # Открытие изображения и конвертация в формат RGB (на случай, если оно в другом формате).
        original_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
        # Преобразование изображения в массив NumPy и нормализация значений пикселей в диапазон [0, 1].
        image_array = np.array(original_image) / 255.0

        # Вызов функции для дополнения изображения и нарезки его на последовательность чанков.
        chunk_sequence, padded_shape, original_shape = pad_and_chunk_image(image_array, CHUNK_SHAPE)

        # В задаче автоэнкодера входная последовательность и целевая последовательность - это одно и то же.
        inputs_sequence = chunk_sequence
        targets_sequence = chunk_sequence

        # Размер входного вектора определяется количеством пикселей в одном "расплющенном" чанке.
        input_size = chunk_sequence.shape[1]
        # Создание объекта модели.
        model = ImageReconstructorRNN(input_size, HIDDEN_SIZE, input_size)

        print(f"Image original size: {original_shape[1]}x{original_shape[0]}")
        print(f"Sequence length: {len(targets_sequence)} chunks")
        print(f"Vector size per chunk: {input_size}")

        best_loss = float('inf')  # Переменная для хранения наилучшего значения потерь.
        patience_counter = 0  # Счетчик эпох без улучшения.
        current_lr = INITIAL_LEARNING_RATE  # Текущий коэффициент обучения.
        # Вычисление "терпения" для планировщика в эпохах на основе процента.
        lr_scheduler_patience = max(1, int(EPOCHS * (LR_PATIENCE_PERCENTAGE / 100.0)))

        print("Starting training...")

        for epoch in range(EPOCHS):
            # Выполнение одного полного шага обучения (forward, backward, update).
            loss = model.train_step(inputs_sequence, targets_sequence, current_lr)

            if epoch % LOG_FREQUENCY == 0 or epoch == EPOCHS - 1:
                print(f'Epoch {epoch}, Loss: {loss:.6f}, LR: {current_lr:.6f}')

            if ENABLE_LR_SCHEDULER:
                # Если текущая ошибка лучше, чем лучшая до этого - сбрасываем счетчик.
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                # Иначе - увеличиваем счетчик "нетерпения".
                else:
                    patience_counter += 1

                # Если счетчик превысил порог "терпения" - снижаем LR.
                if patience_counter >= lr_scheduler_patience:
                    new_lr = current_lr * LR_SCHEDULER_FACTOR
                    if new_lr >= MIN_LEARNING_RATE:
                        print(f"--- No improvement for {lr_scheduler_patience} epochs. Reducing LR to {new_lr:.6f} ---")
                        current_lr = new_lr
                    patience_counter = 0

        print("Training finished. Reconstructing image...")

        # Получение предсказанных чанков от модели.
        reconstructed_chunks_flat = model.forward(inputs_sequence)
        # Преобразование списка выходных векторов в единый массив NumPy.
        reconstructed_chunks = np.squeeze(np.array(reconstructed_chunks_flat))

        # Сборка цельного (но все еще дополненного) изображения из последовательности чанков.
        reconstructed_padded_array = reconstruct_image_from_chunks(reconstructed_chunks, padded_shape, CHUNK_SHAPE)

        # Обрезка лишних пикселей, добавленных на этапе паддинга, для возврата к оригинальному размеру.
        h_orig, w_orig, _ = original_shape
        reconstructed_final_array = reconstructed_padded_array[:h_orig, :w_orig, :]

        # Пост-обработка: ограничение значений в диапазоне [0, 1] и де-нормализация в [0, 255].
        reconstructed_final_array = np.clip(reconstructed_final_array, 0, 1)
        reconstructed_final_array = (reconstructed_final_array * 255).astype(np.uint8)
        # Создание объекта изображения из массива пикселей.
        result_image = Image.fromarray(reconstructed_final_array, 'RGB')

        result_image.save(OUTPUT_IMAGE_PATH)
        print(f"Reconstructed image saved to: {OUTPUT_IMAGE_PATH}")
        result_image.show()

    except FileNotFoundError:
        print(f"Error: File not found at '{INPUT_IMAGE_PATH}'. Please check the path.")
    except Exception as e:
        print(f"An error occurred: {e}")
