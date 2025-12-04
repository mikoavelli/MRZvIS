import numpy as np
from PIL import Image
from model import ImageReconstructorNN
from data_utils import pad_and_chunk_image, reconstruct_image_from_chunks

INPUT_IMAGE_PATH = 'input.bmp'
OUTPUT_IMAGE_PATH = 'output.bmp'

CHUNK_SHAPE = (8, 8)
HIDDEN_SIZE = 64

TRAINING_CONFIG = {
    # ЦЕЛЬ: Сумма квадратичных ошибок за всю эпоху.
    # Так как мы суммируем ошибки тысяч чанков, число будет большим.
    'target_loss': 100.0,

    'max_epochs': 100,  # Эпох нужно меньше, так как обновлений внутри эпохи ОЧЕНЬ много

    # Для чистого SGD (обновление каждый чанк) скорость должна быть очень маленькой,
    # иначе веса "взорвутся".
    'initial_lr': 0.001,

    'log_frequency': 1,  # Логируем каждую эпоху, т.к. они долгие
    'enable_scheduler': True,
    'lr_scheduler_patience_epochs': 5,
    'lr_factor': 0.5,
    'min_lr': 1e-8
}

if __name__ == '__main__':
    try:
        original_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
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
        result_image = Image.fromarray(reconstructed_final_array, 'RGB')
        result_image.save(OUTPUT_IMAGE_PATH)
        print(f"Saved to: {OUTPUT_IMAGE_PATH}")

    except Exception as e:
        print(f"Error: {e}")