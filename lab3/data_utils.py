import numpy as np


def pad_and_chunk_image(image_array, chunk_shape):
    original_shape = image_array.shape
    h, w, c = original_shape
    ch, cw = chunk_shape

    pad_h = (ch - h % ch) % ch
    pad_w = (cw - w % cw) % cw

    if pad_h > 0 or pad_w > 0:
        padded_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode='reflect')
    else:
        padded_array = image_array

    padded_shape = padded_array.shape
    ph, pw, _ = padded_shape

    strides = padded_array.strides
    temp_shape = (ph // ch, ch, pw // cw, cw, c)
    temp_strides = (ch * strides[0], strides[0], cw * strides[1], strides[1], strides[2])

    chunked_view = np.lib.stride_tricks.as_strided(padded_array, shape=temp_shape, strides=temp_strides)
    sequence = chunked_view.transpose(0, 2, 1, 3, 4).reshape(-1, ch * cw * c)

    return sequence, padded_shape, original_shape


def reconstruct_image_from_chunks(sequence, target_shape, chunk_shape):
    h, w, c = target_shape
    ch, cw = chunk_shape

    n_chunks_h = h // ch
    n_chunks_w = w // cw

    chunked_array = sequence.reshape(n_chunks_h, n_chunks_w, ch, cw, c)
    reconstructed_array = chunked_array.transpose(0, 2, 1, 3, 4).reshape(h, w, c)

    return reconstructed_array
