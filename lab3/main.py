import numpy as np
from PIL import Image
from model import ImageReconstructorRNN
from data_utils import pad_and_chunk_image, reconstruct_image_from_chunks

INPUT_IMAGE_PATH = 'input.bmp'
OUTPUT_IMAGE_PATH = 'output.bmp'

CHUNK_SHAPE = (3, 3)
HIDDEN_SIZE = 256
EPOCHS = 500
LOG_FREQUENCY = 1

INITIAL_LEARNING_RATE = 0.001

ENABLE_LR_SCHEDULER = True
LR_SCHEDULER_PATIENCE = 20
LR_SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1e-6

if __name__ == '__main__':
    try:
        original_image = Image.open(INPUT_IMAGE_PATH).convert('RGB')
        image_array = np.array(original_image) / 255.0

        chunk_sequence, padded_shape, original_shape = pad_and_chunk_image(image_array, CHUNK_SHAPE)

        inputs_sequence = chunk_sequence
        targets_sequence = chunk_sequence

        input_size = chunk_sequence.shape[1]

        model = ImageReconstructorRNN(input_size, HIDDEN_SIZE, input_size)

        print(f"Image original size: {original_shape[1]}x{original_shape[0]}")
        print(f"Image padded size: {padded_shape[1]}x{padded_shape[0]}")
        print(f"Sequence length: {len(chunk_sequence)} chunks")
        print(f"Vector size per chunk: {input_size}")
        print("Starting training...")

        best_loss = float('inf')
        patience_counter = 0
        current_lr = INITIAL_LEARNING_RATE

        for epoch in range(EPOCHS):
            loss = model.train_step(inputs_sequence, targets_sequence, current_lr)

            if epoch % LOG_FREQUENCY == 0 or epoch == EPOCHS - 1:
                print(f'Epoch {epoch}, Loss: {loss:.6f}, LR: {current_lr}')

            if ENABLE_LR_SCHEDULER:
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= LR_SCHEDULER_PATIENCE:
                    new_lr = current_lr * LR_SCHEDULER_FACTOR
                    if new_lr >= MIN_LEARNING_RATE:
                        print(
                            f"--- No improvement for {LR_SCHEDULER_PATIENCE} epochs. Reducing learning rate from {current_lr} to {new_lr} ---")
                        current_lr = new_lr
                    patience_counter = 0

        print("Training finished. Reconstructing image...")

        reconstructed_chunks_flat = model.forward(inputs_sequence)
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
