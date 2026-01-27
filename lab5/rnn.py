# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель рекурентной сети с цепью нейросетевых моделей долгой кратковременной памяти
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import config
import numpy as np
from utils import asinh, d_asinh, d_linear, linear


class RNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        np.random.seed(config.RANDOM_SEED)

        # Инициализация весов.
        # Умножение на 0.1 нужно, чтобы веса были маленькими.
        # Если веса большие, нейроны сразу "насыщаются" (выдают большие значения),
        # и сеть перестает учиться.
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1  # Вход -> Скрытый
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1  # Скрытый -> Скрытый (рекурсия)
        self.Why = np.random.randn(output_size, hidden_size) * 0.1  # Скрытый -> Выход

        self.bh = np.zeros((hidden_size, 1))  # Смещение скрытого слоя
        self.by = np.zeros((output_size, 1))  # Смещение выходного слоя

    def forward(self, x, h_prev):
        """
        Прямой проход (Forward Pass).
        Прогоняет последовательность x через сеть.
        """
        T = x.shape[0]
        # h_all нужен для хранения всех промежуточных состояний h_t.
        # Они понадобятся при обратном распространении ошибки (Backward).
        h_all = np.zeros((T, self.hidden_size, 1))

        h_curr = h_prev
        for t in range(T):
            xt = x[t].reshape(-1, 1)

            # Основная формула RNN:
            # h(t) = Activation( Wxh * x(t) + Whh * h(t-1) + bh )
            net = self.Wxh @ xt + self.Whh @ h_curr + self.bh
            h_curr = asinh(net)

            h_all[t] = h_curr  # Сохраняем состояние

        # Вычисление выхода (только для последнего шага последовательности)
        y_pred = linear(self.Why @ h_curr + self.by)
        return y_pred, h_curr, h_all

    def backward(self, x, h_all, h_last, y_pred, y_true, lr):
        """
        Обратное распространение ошибки во времени (BPTT).
        """
        # 1. Вычисление градиента функции потерь MSE.
        # Loss = (y_pred - y_true)^2 (для одного примера Mean = Sum)
        # Производная dLoss/dy = 2 * (y_pred - y_true)
        # Умножаем на 2, чтобы соответствовать строгой формуле производной квадрата.
        error_diff = y_pred - y_true
        dy = 2 * error_diff * d_linear(y_pred)

        # 2. Градиенты выходного слоя
        dWhy = dy @ h_last.T
        dby = dy

        # 3. (Переход в скрытый слой) Градиент, уходящий в скрытый слой
        dh = self.Why.T @ dy

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        dh_next = np.zeros_like(dh)
        T = x.shape[0]

        # 4. Разворачивание во времени назад
        for t in reversed(range(T)):
            xt = x[t].reshape(-1, 1)  # Входные данные, которые сеть видела в момент t
            ht = h_all[t]  # Состояние скрытого слоя, которое было в момент t
            h_prev = h_all[t - 1] if t > 0 else np.zeros_like(ht)  # Состояние на предыдущем шаге

            # Производная сложной функции (chain rule):
            # dLoss/d_net = dLoss/dh * dh/d_net
            # dh/d_net = d_asinh(sinh(h))
            dh_raw = (dh + dh_next) * d_asinh(np.sinh(ht))

            # Накопление градиента функции потерь по матрице весов входного слоя
            dWxh += dh_raw @ xt.T
            # Накопление градиента по матрице рекуррентных весов (связь скрытого слоя с самим собой)
            dWhh += dh_raw @ h_prev.T
            # Накопление градиента по вектору смещения (bias) скрытого слоя
            dbh += dh_raw

            dh_next = self.Whh.T @ dh_raw

            # Обнуляем dh, так как прямой градиент от выхода Y приходит только на последнем шаге T.
            # Для шагов T-1...0 влияние идет только через dh_next.
            dh = np.zeros_like(dh)

        # Gradient Clipping
        for param in [dWxh, dWhh, dbh, dWhy, dby]:
            np.clip(param, -config.CLIP_VALUE, config.CLIP_VALUE, out=param)

        # Обновление весов
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby
