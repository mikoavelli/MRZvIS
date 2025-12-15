# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель рекурентной сети с цепью нейросетевых моделей долгой кратковременной памяти
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичом
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

        # Инициализация весов (масштаб 0.1 для стабильности)
        self.Wxh = np.random.randn(hidden_size, input_size) * 0.1
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.1
        self.Why = np.random.randn(output_size, hidden_size) * 0.1

        self.bh = np.zeros((hidden_size, 1))
        self.by = np.zeros((output_size, 1))

    def forward(self, x, h_prev):
        """
        Прямой проход.
        x: входной вектор (window_size, 1)
        h_prev: начальное скрытое состояние
        """
        T = x.shape[0]
        h_all = np.zeros((T, self.hidden_size, 1))

        h_curr = h_prev
        for t in range(T):
            xt = x[t].reshape(-1, 1)
            # Формула: h = arcsinh(Wxh*x + Whh*h_prev + bh)
            net = self.Wxh @ xt + self.Whh @ h_curr + self.bh
            h_curr = asinh(net)
            h_all[t] = h_curr

        y_pred = linear(self.Why @ h_curr + self.by)
        return y_pred, h_curr, h_all

    def backward(self, x, h_all, h_last, y_pred, y_true, lr):
        """
        Обратное распространение ошибки во времени (BPTT).
        """
        dy = (y_pred - y_true) * d_linear(y_pred)

        dWhy = dy @ h_last.T
        dby = dy
        dh = self.Why.T @ dy

        dWxh = np.zeros_like(self.Wxh)
        dWhh = np.zeros_like(self.Whh)
        dbh = np.zeros_like(self.bh)

        dh_next = np.zeros_like(dh)
        T = x.shape[0]

        for t in reversed(range(T)):
            xt = x[t].reshape(-1, 1)
            ht = h_all[t]
            h_prev = h_all[t - 1] if t > 0 else np.zeros_like(ht)

            # Производная через гиперболический синус:
            # Если h = asinh(net), то net = sinh(h). Производная asinh(net) = d_asinh(net)
            dh_raw = (dh + dh_next) * d_asinh(np.sinh(ht))

            dWxh += dh_raw @ xt.T
            dWhh += dh_raw @ h_prev.T
            dbh += dh_raw

            dh_next = self.Whh.T @ dh_raw

        # Gradient Clipping (защита от взрыва градиентов)
        for param in [dWxh, dWhh, dbh, dWhy, dby]:
            np.clip(param, -config.CLIP_VALUE, config.CLIP_VALUE, out=param)

        # Обновление весов
        self.Wxh -= lr * dWxh
        self.Whh -= lr * dWhh
        self.Why -= lr * dWhy
        self.bh -= lr * dbh
        self.by -= lr * dby
