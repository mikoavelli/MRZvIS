# Индивидуальная лабораторная работа 2 по дисциплине МРЗвИС вариант 6
# "Реализовать модель сети Хэмминга"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import config
import numpy as np


class HammingNetwork:
    def __init__(self):
        self.weights = None
        self.bias = None
        self.labels = []
        self.input_size = 0  # N

    def fit(self, patterns, labels):
        """
        Обучение (инициализация весов).
        В сети Хэмминга обучение — это one-shot процесс: значит просто копируем
        эталоны в матрицу весов.
        """
        # Превращаем в numpy array, если пришел список
        self.weights = np.array(patterns)
        self.labels = labels

        # Определяем размерность входа N по данным
        self.input_size = self.weights.shape[1]

        # Используется для того, чтобы выход первого слоя был положительным
        # и соответствовал мере сходства (расстоянию Хэмминга).
        # Смещение для сети Хэмминга: Bias = N / 2
        self.bias = self.input_size / 2.0

    @staticmethod
    def _activation(x):
        """Функция активации для MAXNET(ReLU - подобная)."""
        return np.maximum(0, x)

    def _maxnet(self, initial_activations):
        """
        Релаксационная сеть (MAXNET).
        Здесь происходит итеративная конкуренция между нейронами.
        """
        M = len(initial_activations)
        if M <= 1:
            return initial_activations, 0

        # Эвристика: epsilon < 1/M.
        # Он должен быть < 1/(M-1), иначе сеть подавит сама себя.
        epsilon = 1.0 / (M + 2.0)

        activations = np.copy(initial_activations)
        prev_activations = np.copy(activations)

        for epoch in range(config.MAX_EPOCHS):
            # Если остался 1 или 0 активных нейронов -> стоп
            # КРИТЕРИЙ РЕЛАКСАЦИИ: Остановка, если остался один активный нейрон.
            if np.sum(activations > 0.0) <= 1:
                return activations, epoch

            sum_prev = np.sum(prev_activations)

            # Латеральное торможение
            # Каждый нейрон подавляет остальных пропорционально своей силе.
            inhibition = sum_prev - prev_activations
            new_activations = prev_activations - epsilon * inhibition

            new_activations = self._activation(new_activations)

            activations = new_activations
            prev_activations = np.copy(activations)

        return activations, config.MAX_EPOCHS

    def predict(self, noisy_pattern):
        """
        Распознавание.
        Двухэтапный процесс: сопоставление (Layer 1) + конкуренция (Layer 2).
        """
        # 1. Слой совпадений: Скалярное произведение векторов.
        # Результат показывает, насколько вход похож на каждый из эталонов.
        dot_prod = np.dot(self.weights, noisy_pattern)
        # Функция активации первого слоя
        layer1_out = 0.5 * dot_prod + self.bias

        # 2. Релаксация (MAXNET) для выбора однозначного победителя.
        final_state, iters = self._maxnet(layer1_out)

        # Обработка исключения: если MAXNET подавил всех, берем максимум из слоя 1.
        if np.max(final_state) == 0:
            winner_idx = np.argmax(layer1_out)
        else:
            winner_idx = np.argmax(final_state)

        return self.labels[winner_idx], iters, winner_idx
