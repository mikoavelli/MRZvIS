# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной  сети с адаптивным коэффициентом обучения с ненормированными весами
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np


class ImageReconstructorNN:
    def __init__(self, input_size, hidden_size, output_size):
        self.inputs = None
        self.hidden_states = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        init_range = 1.0
        self.W_ih = np.random.uniform(-init_range, init_range, (hidden_size, input_size)).astype(np.float32)
        self.W_ho = np.random.uniform(-init_range, init_range, (output_size, hidden_size)).astype(np.float32)

    def forward(self, inputs):
        """Прямой проход."""
        self.inputs = inputs
        # Вход -> Скрытый
        self.hidden_states = np.dot(inputs, self.W_ih.T)
        # Скрытый -> Выход
        outputs = np.dot(self.hidden_states, self.W_ho.T)
        return outputs

    def backward(self, outputs, targets):
        """
        Обратный проход (Pure SGD).
        """
        error = outputs - targets

        # Вычисляем градиенты (Сумма, без усреднения)
        dW_ho = np.dot(error.T, self.hidden_states)
        dh = np.dot(error, self.W_ho)
        dW_ih = np.dot(dh.T, self.inputs)

        return {"W_ih": dW_ih, "W_ho": dW_ho}

    def update_weights_sgd(self, grads, learning_rate):
        """
        Обновление весов: W = W - lr * grad
        """
        self.W_ih -= learning_rate * grads["W_ih"]
        self.W_ho -= learning_rate * grads["W_ho"]

    def train_step(self, inputs, targets, learning_rate):
        """Один шаг обучения (Forward -> Backward -> Update)."""
        outputs = self.forward(inputs)
        grads = self.backward(outputs, targets)
        self.update_weights_sgd(grads, learning_rate)

    def calculate_total_error(self, inputs, targets):
        """
        Отдельный проход для расчета ошибки SSE при замороженных весах.
        """
        total_sse = 0.0
        num_chunks = inputs.shape[0]

        # Проходим циклом, чтобы экономить память
        for i in range(num_chunks):
            chunk_in = inputs[i : i + 1]
            chunk_target = targets[i : i + 1]

            # Локальный forward (без сохранения self.inputs для градиентов)
            hidden = np.dot(chunk_in, self.W_ih.T)
            out = np.dot(hidden, self.W_ho.T)

            diff = out - chunk_target
            total_sse += np.sum(diff**2)

        return total_sse

    def train(self, inputs, targets, config):
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)

        max_epochs = config.get("max_epochs", 1000)
        target_loss = config.get("target_loss", 100.0)

        current_lr = config["initial_lr"]
        min_lr = config.get("min_lr", 1e-9)
        max_lr = config.get("max_lr", 0.01)

        log_freq = config.get("log_frequency", 10)
        enable_scheduler = config.get("enable_scheduler", True)

        # Параметры адаптивности (Bold Driver / Plateau)
        patience_decrease = config.get("lr_patience_decrease", 5)
        factor_decrease = config.get("lr_factor_decrease", 0.5)
        patience_increase = config.get("lr_patience_increase", 3)
        factor_increase = config.get("lr_factor_increase", 1.05)

        best_loss = float("inf")
        wait_counter = 0
        up_counter = 0

        num_chunks = inputs.shape[0]
        print(f"Starting SGD (Weights [-1, 1]). Chunks: {num_chunks}. Init LR: {current_lr}")

        for epoch in range(max_epochs):
            # 1. ЭТАП ОБУЧЕНИЯ (меняем веса)
            indices = np.arange(num_chunks)
            np.random.shuffle(indices)

            for i in indices:
                chunk_in = inputs[i : i + 1]
                chunk_target = targets[i : i + 1]
                # Forward, Backward, Update делаются здесь
                self.train_step(chunk_in, chunk_target, current_lr)

            # 2. ЭТАП ОЦЕНКИ (веса фиксированы)
            epoch_sse = self.calculate_total_error(inputs, targets)

            # Защита от NaN
            if np.isnan(epoch_sse) or np.isinf(epoch_sse):
                print(f"\n[CRITICAL] Loss explosion at epoch {epoch + 1}. Decrease initial_lr!")
                return max_epochs

            # 3. ПРОВЕРКА ЦЕЛИ
            if epoch_sse <= target_loss:
                print(f"\nTarget reached at epoch {epoch + 1}! Loss: {epoch_sse:.2f}")
                return epoch + 1

            status_tag = ""

            # 4. АДАПТИВНОСТЬ
            if enable_scheduler:
                # Порог чувствительности (чтобы не реагировать на шум)
                threshold = 1e-4

                if epoch_sse < best_loss * (1 - threshold):
                    best_loss = epoch_sse
                    wait_counter = 0
                    up_counter += 1

                    if up_counter >= patience_increase:
                        new_lr = current_lr * factor_increase
                        if new_lr <= max_lr:
                            current_lr = new_lr
                            status_tag = f"[UP x{factor_increase}]"
                            up_counter = 0
                        else:
                            current_lr = max_lr
                            status_tag = "[Max LR]"
                else:
                    up_counter = 0
                    wait_counter += 1

                    if wait_counter >= patience_decrease:
                        new_lr = current_lr * factor_decrease
                        if new_lr >= min_lr:
                            current_lr = new_lr
                            status_tag = f"[DOWN x{factor_decrease}]"
                            wait_counter = 0
                            best_loss = epoch_sse
                        else:
                            status_tag = "[Min LR]"

            if (epoch + 1) % log_freq == 0 or epoch == 0 or "UP" in status_tag or "DOWN" in status_tag:
                print(f"Ep {epoch + 1} | SSE: {epoch_sse:.1f} | LR: {current_lr:.8f} {status_tag}")

        return max_epochs


def calculate_compression_z(q, n, secret_layer):
    """
    Расчет коэффициента сжатия по Java-формуле.

    Аргументы:
    q -- количество блоков (chunks count)
    n -- размер блока в элементах (input size, напр. 192)
    secret_layer -- размер скрытого слоя (hidden size)
    """

    # Числитель: Исходный размер в битах (8 бит на канал пикселя)
    numerator = 8.0 * q * n

    # Знаменатель:
    # 2 * 32.0 -> заголовки (ширина/высота)
    # (n + q) * secret_layer * 64.0 -> Веса декодера + Сжатые данные (в double/64 bit)
    denominator = 2 * 32.0 + (n + q) * secret_layer * 64.0

    return numerator / denominator
