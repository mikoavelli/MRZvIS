# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной  сети с адаптивным коэффициентом обучения с ненормированными весами
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np
from data_utils import pad_and_chunk_image, reconstruct_image_from_chunks
from model import ImageReconstructorNN
from PIL import Image

INPUT_IMAGE_PATH = "input2.bmp"
OUTPUT_IMAGE_PATH = "output.bmp"

CHUNK_SHAPE = (8, 8)
HIDDEN_SIZE = 64

TRAINING_CONFIG = {
    "target_loss": 3000.0,
    "max_epochs": 1000,
    "initial_lr": 0.0001,
    "log_frequency": 1,
    "enable_scheduler": True,
    "lr_patience_decrease": 3,
    "lr_factor_decrease": 0.7,
    "lr_patience_increase": 3,
    "lr_factor_increase": 1.05,
    "min_lr": 1e-7,
}

if __name__ == "__main__":
    try:
        original_image = Image.open(INPUT_IMAGE_PATH).convert("RGB")
        image_array = np.array(original_image) / 255.0

        chunk_sequence, padded_shape, original_shape = pad_and_chunk_image(image_array, CHUNK_SHAPE)
        input_size = chunk_sequence.shape[1]

        model = ImageReconstructorNN(input_size, HIDDEN_SIZE, input_size)

        print(f"Image original size: {original_shape[1]}x{original_shape[0]}")
        print(f"Total chunks to process per epoch: {chunk_sequence.shape[0]}")

        # Запуск обучения
        model.train(chunk_sequence, chunk_sequence, TRAINING_CONFIG)

        print("Training finished. Reconstructing...")

        # Inference
        reconstructed_chunks_flat = model.forward(chunk_sequence.astype(np.float32))
        reconstructed_chunks = np.squeeze(np.array(reconstructed_chunks_flat))

        reconstructed_padded_array = reconstruct_image_from_chunks(reconstructed_chunks, padded_shape, CHUNK_SHAPE)
        h_orig, w_orig, _ = original_shape
        reconstructed_final_array = reconstructed_padded_array[:h_orig, :w_orig, :]

        reconstructed_final_array = np.clip(reconstructed_final_array, 0, 1)
        reconstructed_final_array = (reconstructed_final_array * 255).astype(np.uint8)
        result_image = Image.fromarray(reconstructed_final_array, "RGB")
        result_image.save(OUTPUT_IMAGE_PATH)
        print(f"Saved to: {OUTPUT_IMAGE_PATH}")

    except Exception as e:
        print(f"Error: {e}")
