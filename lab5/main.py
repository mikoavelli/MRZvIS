# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель рекурентной сети с цепью нейросетевых моделей долгой кратковременной памяти
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import config
import numpy as np
from rnn import RNN
from tabulate import tabulate
from utils import DataProcessor, FibonacciStrategy


def run_pipeline():
    strategy = FibonacciStrategy()
    print(f"Запуск эксперимента: {strategy.__class__.__name__}")

    # 2. Генерация данных
    # Получаем [1, 1, 2, 3, 5, ..., 12586269025]
    history_raw, future_raw = strategy.generate(config.SEQ_LEN, config.FUTURE_STEPS)

    # 3. Подготовка данных
    processor = DataProcessor(config.WINDOW_SIZE)
    # Показываем таблицу с ОРИГИНАЛЬНЫМИ числами (для отчета)
    processor.print_training_table(history_raw, config.WINDOW_SIZE)

    # Числа Фибоначчи растут экспоненциально. Сеть не может предсказать число 10^9,
    # если училась на числах 1..100, так как веса не могут вырасти так быстро.
    # Логарифмирование. Экспонента превращается в линию.
    # ln(1, 2, 3, 5, 8...) -> (0, 0.69, 1.1, 1.6, 2.0...)
    history_log = np.log(history_raw)

    # Нормализация (Z-score) логарифмированных данных, чтобы загнать их в диапазон около [-2, 2]
    history_log_norm = processor.normalize(history_log)

    # Нарезка на окна
    X, Y = processor.create_windows(history_log_norm)

    # 4. Инициализация сети
    model = RNN(config.INPUT_SIZE, config.HIDDEN_SIZE, config.OUTPUT_SIZE)

    # 5. Обучение
    print(f"\nНачало обучения ({config.EPOCHS} эпох) на логарифмированных данных...")
    for epoch in range(1, config.EPOCHS + 1):
        total_loss = 0

        # h_init = 0. Мы сбрасываем контекст для каждого примера (окна).
        # Это значит, что сеть учится предсказывать Y только по окну X,
        # не используя память о предыдущих окнах (stateless training).
        h_init = np.zeros((config.HIDDEN_SIZE, 1))

        for i in range(len(X)):
            # Прямой проход
            # ВОТ ЗДЕСЬ ПРОИСХОДИТ ОБНУЛЕНИЕ (перед подачей в forward)
            # Мы подаем h_init, который был инициализирован нулями выше
            y_pred, _, h_all = model.forward(X[i], h_init)

            # Ошибка
            loss = (y_pred - Y[i].reshape(1, 1)) ** 2
            total_loss += loss.item()

            # Обратный проход (обучение)
            model.backward(X[i], h_all, h_all[-1], y_pred, Y[i].reshape(1, 1), config.LEARNING_RATE)

        if epoch % 100 == 0:
            print(f"Эпоха {epoch}/{config.EPOCHS}, Loss: {total_loss / len(X):.6f}")

    # 6. Прогнозирование (Авторегрессия) -- Обнуление
    print("\nПрогнозирование...")
    # Берем последнее известное окно (в нормализованном логарифмическом виде)
    curr_window = history_log_norm[-config.WINDOW_SIZE :].reshape(config.WINDOW_SIZE, 1)
    preds_log_norm = []

    # Сбрасываем контекст перед прогнозом
    h_context = np.zeros((config.HIDDEN_SIZE, 1))

    for _ in range(config.FUTURE_STEPS):
        # 1. Предсказываем одно число вперед
        y_out, _, _ = model.forward(curr_window, h_context)
        val = y_out.item()
        preds_log_norm.append(val)

        # 2. Сдвигаем окно: удаляем самое старое число, добавляем предсказанное
        # "авторегрессия" — выход становится входом.
        curr_window = np.vstack([curr_window[1:], [[val]]])

    # 7. Обратное преобразование
    # Шаг 1: Денормализация
    preds_log = processor.denormalize(np.array(preds_log_norm))

    # Шаг 2: Экспонента (восстанавливаем числа Фибоначчи)
    preds_original = np.exp(preds_log)

    # 8. Вывод и сравнение с реальными будущими значениями
    diff = preds_original - future_raw
    table_res = [
        ["Реальное"] + [f"{v:.2f}" for v in future_raw],
        ["Прогноз"] + [f"{v:.2f}" for v in preds_original],
        ["Разница"] + [f"{v:.2f}" for v in diff],
    ]
    print("\n=== Результаты прогноза (Оригинальные числа) ===")
    print(tabulate(table_res, tablefmt="fancy_grid"))

    # 9. Графики
    # graphs.plot_prediction(history_raw, future_raw, preds_original, strategy.__class__.__name__)
    # graphs.build_benchmark_graphs()


if __name__ == "__main__":
    run_pipeline()
