# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичем
# Файл реализации линейной рециркуляционной сети
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач. Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np


class ImageReconstructorNN:
    """
    Инициализация сети и создание весовых матриц.
    """

    def __init__(self, input_size, hidden_size, output_size):
        self.inputs = None
        self.hidden_states = None
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # --- 1. ИНИЦИАЛИЗАЦИЯ ВЕСОВ ---
        # ТРЕБОВАНИЕ ТЗ: Равномерное распределение в диапазоне [-1, 1].
        init_range = 1.0

        # ОПТИМИЗАЦИЯ: Принудительный перевод в тип float32.
        # Стандартный float64 избыточен для нейросетей. float32 ускоряет вычисления
        # в 1.5-2 раза за счет использования векторных инструкций процессора (AVX)
        # и занимает в 2 раза меньше памяти.
        self.W_ih = np.random.uniform(-init_range, init_range, (hidden_size, input_size)).astype(np.float32)
        self.W_ho = np.random.uniform(-init_range, init_range, (output_size, hidden_size)).astype(np.float32)

        # --- 2. ИНИЦИАЛИЗАЦИЯ ОПТИМИЗАТОРА ADAM ---
        # Adam требует хранения "истории" градиентов для каждого веса.
        # m - первый момент (среднее значение градиента / инерция).
        # v - второй момент (среднее значение квадрата градиента / дисперсия).
        self.m, self.v = {}, {}
        for param in ['W_ih', 'W_ho']:
            # Создаем буферы нулей той же формы, что и матрицы весов
            self.m[param] = np.zeros_like(getattr(self, param))
            self.v[param] = np.zeros_like(getattr(self, param))

            # Гиперпараметры Adam (стандартные значения)
            self.beta1 = 0.9  # Коэффициент "инерции" (как долго помнить направление движения)
            self.beta2 = 0.999  # Коэффициент для масштабирования шага
            self.epsilon = 1e-8  # Защита от деления на ноль
            self.t = 0  # Счетчик шагов (для коррекции старта)

    def forward(self, inputs):
        """
        Прямой проход (Forward Pass).
        Преобразует входные данные в выходные через скрытый слой.
        Использует ВЕКТОРИЗАЦИЮ: обрабатывает все чанки одновременно.
        """
        # Сохраняем вход для вычисления градиентов (backward)
        self.inputs = inputs

        # --- ЭТАП 1: КОДИРОВАНИЕ (Вход -> Скрытый слой) ---
        # Умножение матрицы входов (N, 192) на транспонированную матрицу весов (192, 64).
        # Результат: матрица (N, 64), где N - количество чанков.
        # Это "сжатое" представление изображения.
        self.hidden_states = np.dot(inputs, self.W_ih.T)

        # --- ЭТАП 2: ДЕКОДИРОВАНИЕ (Скрытый слой -> Выход) ---
        # Умножение скрытых состояний (N, 64) на матрицу весов декодера (64, 192).
        # Результат: матрица (N, 192).
        # Это восстановленные чанки изображения.
        outputs = np.dot(self.hidden_states, self.W_ho.T)

        return outputs

    def backward(self, outputs, targets, compute_loss=False):
        """
        Обратный проход (Backward Pass).
        Вычисляет ошибку и градиенты для обновления весов.
        """
        N = self.inputs.shape[0]  # Количество примеров (чанков)
        # 1. Вычисляем вектор ошибки для каждого пикселя каждого чанка
        error = outputs - targets

        # Расчет MSE (Среднеквадратичной ошибки).
        total_loss = 0.0
        if compute_loss:
            # np.mean усредняет по всем элементам матрицы
            total_loss = np.mean(error ** 2) * self.input_size

        # Градиент для весов W_ho (Скрытый -> Выход)
        # Формула: dL/dW = Error^T * Input_of_layer
        dW_ho = np.dot(error.T, self.hidden_states)

        # "Протаскивание" ошибки назад через веса W_ho на скрытый слой
        dh = np.dot(error, self.W_ho)

        # Градиент для весов W_ih (Вход -> Скрытый)
        dW_ih = np.dot(dh.T, self.inputs)

        return {  # Возвращаем усредненные градиенты (делим на количество чанков N)
            'W_ih': dW_ih / N,
            'W_ho': dW_ho / N
        }, total_loss

    def update_weights_adam(self, grads, learning_rate):
        """
        Обновление весов методом Adam (Adaptive Moment Estimation).
        Реализует адаптивный коэффициент обучения.
        2"""
        self.t += 1

        # Предвычисляем множители для коррекции смещения момента (bias correction).
        # Это статистическая коррекция для старта обучения)
        correction1 = 1 - self.beta1 ** self.t
        correction2 = 1 - self.beta2 ** self.t

        for key in ['W_ih', 'W_ho']:
            # Получаем ссылки на текущие параметры
            param = getattr(self, key)
            grad = grads[key]

            m = self.m[key]
            v = self.v[key]

            # 1. Обновляем моментум (направление движения)
            # m = beta1 * m + (1 - beta1) * g
            m[:] = self.beta1 * m + (1 - self.beta1) * grad

            # 2. Обновляем адаптивную часть (оценка "шумности" градиента)
            # v = beta2 * v + (1 - beta2) * g^2
            v[:] = self.beta2 * v + (1 - self.beta2) * (grad ** 2)

            # 3. Корректируем смещение старта (чтобы не стартовать с нуля)
            m_corr = m / correction1
            v_corr = v / correction2

            # 4. ФИНАЛЬНОЕ ОБНОВЛЕНИЕ ВЕСА
            # Вес = Вес - LR * Моментум / (Корень(Дисперсия) + epsilon)
            param -= learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def train_step(self, inputs, targets, learning_rate, compute_loss=False):
        outputs = self.forward(inputs)
        grads, loss = self.backward(outputs, targets, compute_loss)
        self.update_weights_adam(grads, learning_rate)
        return loss

    def train(self, inputs, targets, config):
        """
        Полный цикл обучения.
        Управляет эпохами, логами и планировщиком скорости обучения.
        """
        # --- ПОДГОТОВКА ДАННЫХ ---
        # Конвертируем входные данные в float32 один раз перед циклом для скорости.
        inputs = inputs.astype(np.float32)
        targets = targets.astype(np.float32)

        # Извлекаем настройки
        max_epochs = config.get('max_epochs', 10000)
        target_loss = config.get('target_loss', 0.0)  # Цель, к которой стремимся
        initial_lr = config['initial_lr']
        log_frequency = config['log_frequency']

        # Настройки планировщика (Scheduler)
        enable_scheduler = config.get('enable_scheduler', False)
        patience_threshold = config.get('lr_scheduler_patience_epochs', 20)
        factor = config['lr_factor']
        min_lr = config['min_lr']

        # Переменные состояния обучения
        patience_counter = 0  # Сколько эпох мы ожидаем улучшения
        best_loss = float('inf')  # Лучшая ошибка, которую мы видели
        loss = float('inf')
        current_lr = initial_lr

        print(f"Starting optimized training (float32). Target Loss: {target_loss}...")

        # --- ГЛАВНЫЙ ЦИКЛ ПО ЭПОХАМ ---
        for epoch in range(max_epochs):
            # Шаг обучения. compute_loss=True, чтобы мы могли проверить условие выхода.
            loss = self.train_step(inputs, targets, current_lr, compute_loss=True)

            # 1. ПРОВЕРКА ЦЕЛИ
            # Если ошибка стала меньше целевой - победа, выходим.
            if loss <= target_loss:
                print(f"\nTarget loss {target_loss} reached at epoch {epoch}! Loss: {loss:.6f}")
                break

            # 2. ЛОГИРОВАНИЕ
            if epoch % log_frequency == 0:
                print(f'Epoch {epoch}, Loss: {loss:.6f}, LR: {current_lr:.6f}')

            # 3. ПЛАНИРОВЩИК (SCHEDULER)
            if enable_scheduler:
                # Если текущая ошибка лучше рекорда - обновляем рекорд и сбрасываем счетчик
                if loss < best_loss:
                    best_loss = loss
                    patience_counter = 0
                else:
                    # Если рекорда нет - копим "терпение"
                    patience_counter += 1

                # Если терпение лопнуло (слишком долго нет улучшений)
                if patience_counter >= patience_threshold:
                    new_lr = current_lr * factor
                    # Снижаем скорость, если не достигли минимума
                    if new_lr >= min_lr:
                        print(f"--- No new best loss for {patience_threshold} epochs. Reducing LR to {new_lr:.6f} ---")
                        current_lr = new_lr
                        patience_counter = 0
        else:
            # Сработает, если цикл закончился сам по себе (достиг max_epochs)
            print(f"\nMax epochs ({max_epochs}) reached. Final Loss: {loss:.6f}")
