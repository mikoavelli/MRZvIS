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

CHUNK_SHAPE = (8, 8)
HIDDEN_SIZE = 64

TRAINING_CONFIG = {
    'target_loss': 0.1,
    'max_epochs': 50000,
    'initial_lr': 0.005,
    'log_frequency': 50,
    'enable_scheduler': True,
    'lr_scheduler_patience_epochs': 50,
    'lr_factor': 0.5,
    'min_lr': 1e-6
}

if __name__ == '__main__':
    try:
        original_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
        image_array = np.array(original_image) / 255.0

        chunk_sequence, padded_shape, original_shape = pad_and_chunk_image(image_array, CHUNK_SHAPE)

        input_size = chunk_sequence.shape[1]
        model = ImageReconstructorNN(input_size, HIDDEN_SIZE, input_size)

        print(f"Image original size: {original_shape[1]}x{original_shape[0]}")
        print(f"Sequence length: {len(chunk_sequence)} chunks")
        print(f"Vector size per chunk: {input_size}")
        print(f"Hidden layer size (bottleneck): {HIDDEN_SIZE}")

        model.train(chunk_sequence, chunk_sequence, TRAINING_CONFIG)

        print("Training finished. Reconstructing image...")

        reconstructed_chunks_flat = model.forward(chunk_sequence)
        reconstructed_chunks = np.squeeze(np.array(reconstructed_chunks_flat))

        reconstructed_padded_array = reconstruct_image_from_chunks(reconstructed_chunks, padded_shape, CHUNK_SHAPE)

        h_orig, w_orig, _ = original_shape
        reconstructed_final_array = reconstructed_padded_array[:h_orig, :w_orig, :]

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
