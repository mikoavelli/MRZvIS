# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 6
# "Реализовать модель сети Хэмминга"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичом
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач. Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

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
        patterns: список векторов (или numpy array).
        """
        # Превращаем в numpy array, если пришел список
        self.weights = np.array(patterns)
        self.labels = labels

        # Определяем размерность входа N по данным
        self.input_size = self.weights.shape[1]

        # Смещение для сети Хэмминга: Bias = N / 2
        self.bias = self.input_size / 2.0

    @staticmethod
    def _activation(x):
        return np.maximum(0, x)

    def _maxnet(self, initial_activations):
        M = len(initial_activations)
        if M <= 1:
            return initial_activations, 0

        # Эвристика: epsilon < 1/M.
        epsilon = 1.0 / (M + 2.0)

        activations = np.copy(initial_activations)
        prev_activations = np.copy(activations)

        for epoch in range(config.MAX_EPOCHS):
            # Если остался 1 или 0 активных нейронов -> стоп
            if np.sum(activations > 0.01) <= 1:
                return activations, epoch

            sum_prev = np.sum(prev_activations)

            # Латеральное торможение
            inhibition = sum_prev - prev_activations
            new_activations = prev_activations - epsilon * inhibition

            new_activations = self._activation(new_activations)

            if np.allclose(new_activations, prev_activations, atol=1e-5):
                return new_activations, epoch

            activations = new_activations
            prev_activations = np.copy(activations)

        return activations, config.MAX_EPOCHS

    def predict(self, noisy_pattern):
        # 1. Слой совпадений
        dot_prod = np.dot(self.weights, noisy_pattern)
        layer1_out = 0.5 * dot_prod + self.bias

        # 2. MAXNET
        final_state, iters = self._maxnet(layer1_out)

        if np.max(final_state) == 0:
            winner_idx = np.argmax(layer1_out)
        else:
            winner_idx = np.argmax(final_state)

        return self.labels[winner_idx], iters, winner_idx
