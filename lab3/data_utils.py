# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной  сети с адаптивным коэффициентом обучения с ненормированными весами
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np


def pad_and_chunk_image(image_array, chunk_shape):
    """
    Подготавливает изображение для подачи в нейросеть.
    1. Дополняет края (padding), чтобы изображение делилось на чанки без остатка.
    2. Нарезает изображение на квадратики (чанки) и выпрямляет их в векторы.
    """
    original_shape = image_array.shape
    h, w, c = original_shape
    ch, cw = chunk_shape

    # Вычисляем, сколько пикселей не хватает по высоте и ширине.
    pad_h = (ch - h % ch) % ch
    pad_w = (cw - w % cw) % cw

    # Если дополнение необходимо, создаем новый массив с отраженными по краям пикселями.
    if pad_h > 0 or pad_w > 0:
        padded_array = np.pad(image_array, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")
    else:
        padded_array = image_array

    # Создаём "вид" массива в виде чанков, не копируя данные.
    padded_shape = padded_array.shape
    ph, pw, _ = padded_shape

    # Новая форма и "шаги" для 5D-представления.
    strides = padded_array.strides
    temp_shape = (ph // ch, ch, pw // cw, cw, c)
    temp_strides = (
        ch * strides[0],
        strides[0],
        cw * strides[1],
        strides[1],
        strides[2],
    )

    # Создание "вида" (view) массива без копирования данных.
    chunked_view = np.lib.stride_tricks.as_strided(padded_array, shape=temp_shape, strides=temp_strides)
    # Преобразование 5D-вида в 2D-массив, где каждая строка - это один "расплющенный" чанк.
    sequence = chunked_view.transpose(0, 2, 1, 3, 4).reshape(-1, ch * cw * c)

    return sequence, padded_shape, original_shape


def reconstruct_image_from_chunks(sequence, target_shape, chunk_shape):
    h, w, c = target_shape
    ch, cw = chunk_shape

    # Количество чанков по высоте и ширине.
    n_chunks_h = h // ch
    n_chunks_w = w // cw

    # Преобразование плоской последовательности обратно в 5D-массив чанков.
    chunked_array = sequence.reshape(n_chunks_h, n_chunks_w, ch, cw, c)
    # Сборка чанков в единый 3D-массив изображения.
    reconstructed_array = chunked_array.transpose(0, 2, 1, 3, 4).reshape(h, w, c)

    return reconstructed_array
