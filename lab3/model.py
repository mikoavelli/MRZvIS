# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной сети с адаптивным коэффициентом обучения с ненормированными весами"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичем
# Файл реализации линейной рециркуляционной сети
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач. Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import numpy as np


class ImageReconstructorRNN:
    """
    Класс, реализующий рекуррентную нейронную сеть (РНН) для задачи
    реконструкции изображений (автоэнкодер).
    """

    def __init__(self, input_size, hidden_size, output_size):
        """
        Конструктор класса. Инициализирует все обучаемые параметры (веса)
        и переменные для оптимизатора Adam.
        """
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Матрица: Вход -> Скрытый слой
        self.W_ih = np.random.uniform(-0.1, 0.1, (hidden_size, input_size))
        # Матрица: Скрытый слой -> Скрытый слой (рекуррентная связь)
        self.W_hh = np.random.uniform(-0.1, 0.1, (hidden_size, hidden_size))
        # Матрица: Скрытый слой -> Выход
        self.W_ho = np.random.uniform(-0.1, 0.1, (output_size, hidden_size))

        # Веса-смещения (биасы) инициализируются нулями.
        self.b_h = np.zeros((hidden_size, 1))  # Вектор смещения для скрытого слоя
        self.b_o = np.zeros((output_size, 1))  # Вектор смещения для выходного слоя

        # 'm' хранит скользящее среднее градиентов (моментум).
        # 'v' хранит скользящее среднее квадратов градиентов (адаптивная часть).
        self.m, self.v = {}, {}
        for param in ['W_ih', 'W_hh', 'W_ho', 'b_h', 'b_o']:
            self.m[param] = np.zeros_like(getattr(self, param))
            self.v[param] = np.zeros_like(getattr(self, param))

        # Стандартные гиперпараметры для Adam.
        # self.beta1: Коэффициент затухания для "момента" (скользящего среднего градиентов).
        # Отвечает за инерцию обновлений, помогает ускоряться в правильном направлении.
        self.beta1 = 0.9
        # self.beta2: Коэффициент затухания для скользящего среднего квадратов градиентов.
        # Отвечает за адаптацию скорости обучения для каждого веса индивидуально.
        self.beta2 = 0.999
        # self.epsilon: Очень маленькое число для предотвращения деления на ноль в формуле Adam.
        # Обеспечивает численную стабильность.
        self.epsilon = 1e-8
        # self.t: Счетчик шагов обучения. Используется для коррекции смещения 'm' и 'v'
        # на начальных этапах, когда они инициализированы нулями.
        self.t = 0

    def forward(self, inputs):
        """
        Прямой проход. Выполняет предсказание для всей последовательности.
        """
        self.inputs = inputs
        # Инициализация "нулевого" скрытого состояния (памяти до начала последовательности).
        self.hidden_states = {-1: np.zeros((self.hidden_size, 1))}
        outputs = []
        # Цикл по всем чанкам во входной последовательности.
        for i in range(len(inputs)):
            x = inputs[i].reshape(-1, 1)
            # Новое скрытое состояние вычисляется на основе входа и предыдущего состояния.
            # Функция активации отсутствует, что делает эту операцию линейной.
            self.hidden_states[i] = (
                    np.dot(self.W_ih, x) +
                    np.dot(self.W_hh, self.hidden_states[i - 1]) +
                    self.b_h
            )
            # Выходное значение генерируется на основе нового скрытого состояния.
            output = np.dot(self.W_ho, self.hidden_states[i]) + self.b_o
            outputs.append(output)
        return outputs

    def backward(self, outputs, targets):
        """
        Обратный проход (Backpropagation Through Time). Вычисляет градиенты (производные)
        функции потерь по всем весам.
        """
        # Инициализация градиентов нулями.
        dW_ih, dW_hh, dW_ho = np.zeros_like(self.W_ih), np.zeros_like(self.W_hh), np.zeros_like(self.W_ho)
        db_h, db_o = np.zeros_like(self.b_h), np.zeros_like(self.b_o)
        # Градиент скрытого состояния, передаваемый на предыдущий шаг по времени.
        dh_next = np.zeros_like(self.hidden_states[0])
        total_loss = 0

        # Цикл по последовательности в ОБРАТНОМ порядке.
        for i in reversed(range(len(self.inputs))):
            # 1. Вычисление ошибки и потерь (MSE) для текущего шага.
            error = outputs[i] - targets[i].reshape(-1, 1)
            total_loss += np.sum(error ** 2)

            # 2. Расчет градиентов для выходного слоя.
            dW_ho += np.dot(error, self.hidden_states[i].T)
            db_o += error

            # 3. Распространение ошибки назад: от выхода к скрытому слою.
            dh = np.dot(self.W_ho.T, error) + dh_next

            # Для линейной функции активации ее производная равна 1, поэтому dh_raw = dh.
            dh_raw = dh

            # 4. Расчет градиентов для скрытого и входного слоев на основе распространенной ошибки.
            dW_hh += np.dot(dh_raw, self.hidden_states[i - 1].T)
            dW_ih += np.dot(dh_raw, self.inputs[i].reshape(1, -1))
            db_h += dh_raw

            # 5. Обновление градиента для передачи на следующий (предыдущий по времени) шаг.
            dh_next = np.dot(self.W_hh.T, dh_raw)

        # Ограничение градиентов (Clipping) для предотвращения "взрыва градиентов".
        for dparam in [dW_ih, dW_hh, dW_ho, db_h, db_o]:
            np.clip(dparam, -5, 5, out=dparam)

        # Возвращаем словарь с градиентами и среднее значение потерь.
        return {'W_ih': dW_ih, 'W_hh': dW_hh, 'W_ho': dW_ho, 'b_h': db_h, 'b_o': db_o}, total_loss / len(targets)

    def update_weights_adam(self, grads, learning_rate):
        self.t += 1
        for key in ['W_ih', 'W_hh', 'W_ho', 'b_h', 'b_o']:
            param = getattr(self, key)
            # Вычисление скользящих средних для градиентов и их квадратов.
            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[key]
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[key] ** 2)
            # Коррекция смещения для начальных шагов обучения.
            m_corr = self.m[key] / (1 - self.beta1 ** self.t)
            v_corr = self.v[key] / (1 - self.beta2 ** self.t)
            # Финальное обновление веса.
            param -= learning_rate * m_corr / (np.sqrt(v_corr) + self.epsilon)

    def train_step(self, inputs, targets, learning_rate):
        """
        Один полный шаг обучения.
        """
        outputs = self.forward(inputs)
        grads, loss = self.backward(outputs, targets)
        self.update_weights_adam(grads, learning_rate)
        return loss
