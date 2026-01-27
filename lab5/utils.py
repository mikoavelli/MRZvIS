# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель рекурентной сети с цепью нейросетевых моделей долгой кратковременной памяти
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np
from tabulate import tabulate


def asinh(x):
    """Функция активации: Гиперболический арксинус."""
    return np.arcsinh(x)


def d_asinh(x):
    """
    Производная функции asinh(x).
    Формула: f'(x) = 1 / sqrt(1 + x^2).
    """
    return 1.0 / np.sqrt(1.0 + np.square(x))


def linear(x):
    """Линейная активация для выходного слоя (регрессия)."""
    return x


def d_linear(x):
    """Производная линейной функции = 1."""
    return np.ones_like(x)


# === СТРАТЕГИИ ГЕНЕРАЦИИ ===
class BaseStrategy:
    def generate(self, n_steps, n_future):
        pass


class TrigonometryStrategy(BaseStrategy):
    def generate(self, n_steps, n_future):
        t = np.arange(n_steps + n_future)
        full = np.sin(0.3 * t)  # Синусоида с частотой 0.3
        return full[:n_steps], full[n_steps:]


class FibonacciStrategy(BaseStrategy):
    def generate(self, n_steps, n_future):
        """
        Генерирует последовательность Фибоначчи.
        Важно: генерируем сразу (n_steps + n_future) элементов,
        чтобы иметь эталон (future_raw) для сравнения прогноза.
        """
        seq = [1.0, 1.0]
        count = n_steps + n_future
        for _ in range(count - 2):
            seq.append(seq[-1] + seq[-2])
        return np.array(seq[:n_steps]), np.array(seq[n_steps:])


class GeometricStrategy(BaseStrategy):
    def generate(self, n_steps, n_future):
        full = [1.0 * (0.5**i) for i in range(n_steps + n_future)]
        return np.array(full[:n_steps]), np.array(full[n_steps:])


class ArithmeticStrategy(BaseStrategy):
    def generate(self, n_steps, n_future):
        full = [float(i) for i in range(n_steps + n_future)]
        return np.array(full[:n_steps]), np.array(full[n_steps:])


# === РАБОТА С ДАННЫМИ ===
class DataProcessor:
    def __init__(self, window_size):
        self.window_size = window_size
        self.mean = 0.0
        self.std = 1.0

    def normalize(self, data):
        """
        Z-score нормализация: (x - mean) / std.
        Приводит данные к распределению с центром в 0 и разбросом 1.
        Это критически важно для эффективной работы градиентного спуска.
        """
        self.mean = np.mean(data)
        self.std = np.std(data)
        return (data - self.mean) / self.std

    def denormalize(self, data):
        """Обратное преобразование для получения реальных значений."""
        return data * self.std + self.mean

    def create_windows(self, data):
        """
        Метод скользящего окна.
        Превращает одномерный ряд [1, 2, 3, 4, 5, 6] с окном 3 в:
        X: [[1,1,2], [1,2,3], [2,3,5]]
        Y: [[3],     [5],     [8]]
        Reshape нужен для совместимости с матричными операциями в RNN.
        """
        X, Y = [], []
        for i in range(len(data) - self.window_size):
            X.append(data[i : i + self.window_size])
            Y.append(data[i + self.window_size])
        # Возвращаем тензоры размерности (Batch_Size, Window_Size, Features=1)
        return np.array(X)[:, :, None], np.array(Y)[:, None]

    @staticmethod
    def print_training_table(data, window_size):
        # Убрали аргумент limit и срез [:limit] в цикле
        table = []
        headers = [f"x[t-{window_size - i}]" for i in range(window_size)] + ["y"]

        # Теперь выводим ВСЕ строки
        for i in range(len(data) - window_size):
            row = data[i : i + window_size].tolist() + [data[i + window_size]]
            table.append(row)

        print("\n=== Полная обучающая выборка ===")
        print(tabulate(table, headers=headers, tablefmt="fancy_grid", floatfmt=".1f"))
