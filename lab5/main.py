# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель рекурентной сети с цепью нейросетевых моделей долгой кратковременной памяти
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы 221702 БГУИР Целуйко Дмитрием Александровичом
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import config
import graphs
import numpy as np
from rnn import RNN
from tabulate import tabulate
from utils import DataProcessor, FibonacciStrategy


def run_pipeline():
    # ==========================================
    # ЧАСТЬ 1: ЭКСПЕРИМЕНТ С ФИБОНАЧЧИ
    # ==========================================
    strategy = FibonacciStrategy()
    print(f"Запуск эксперимента: {strategy.__class__.__name__}")

    # 1. Генерация данных
    history_raw, future_raw = strategy.generate(config.SEQ_LEN, config.FUTURE_STEPS)

    # 2. Вывод таблицы (Оригинальные числа)
    processor = DataProcessor(config.WINDOW_SIZE)
    processor.print_training_table(history_raw, config.WINDOW_SIZE)

    # 3. Преобразование (Log -> Normalize)
    history_log = np.log(history_raw)
    history_log_norm = processor.normalize(history_log)
    X, Y = processor.create_windows(history_log_norm)

    # 4. Инициализация сети
    model = RNN(
        input_size=config.INPUT_SIZE,
        hidden_size=config.HIDDEN_SIZE,
        output_size=config.OUTPUT_SIZE,
    )

    # 5. Обучение
    print(f"\nНачало обучения ({config.EPOCHS} эпох) на логарифмированных данных...")
    for epoch in range(1, config.EPOCHS + 1):
        total_loss = 0
        h_init = np.zeros((config.HIDDEN_SIZE, 1))

        for i in range(len(X)):
            y_pred, _, h_all = model.forward(X[i], h_init)
            loss = (y_pred - Y[i].reshape(1, 1)) ** 2
            total_loss += loss.item()
            model.backward(X[i], h_all, h_all[-1], y_pred, Y[i].reshape(1, 1), config.LEARNING_RATE)

        if epoch % 100 == 0:
            print(f"Эпоха {epoch}/{config.EPOCHS}, Loss: {total_loss / len(X):.6f}")

    # 6. Прогноз
    print("\nПрогнозирование...")
    curr_window = history_log_norm[-config.WINDOW_SIZE :].reshape(config.WINDOW_SIZE, 1)
    preds_log_norm = []
    h_context = np.zeros((config.HIDDEN_SIZE, 1))

    for _ in range(config.FUTURE_STEPS):
        y_out, _, _ = model.forward(curr_window, h_context)
        val = y_out.item()
        preds_log_norm.append(val)
        curr_window = np.vstack([curr_window[1:], [[val]]])

    # 7. Обратное преобразование (Denormalize -> Exp)
    preds_log = processor.denormalize(np.array(preds_log_norm))
    preds_original = np.exp(preds_log)

    # 8. Вывод результатов
    diff = preds_original - future_raw
    table_res = [
        ["Реальное"] + [f"{v:.2f}" for v in future_raw],
        ["Прогноз"] + [f"{v:.2f}" for v in preds_original],
        ["Разница"] + [f"{v:.2f}" for v in diff],
    ]
    print("\n=== Результаты прогноза (Оригинальные числа) ===")
    print(tabulate(table_res, tablefmt="fancy_grid"))

    # 9. Построение графика прогноза Фибоначчи
    # Программа остановится здесь, пока вы не закроете окно с графиком
    graphs.plot_prediction(history_raw, future_raw, preds_original, strategy.__class__.__name__)

    # ==========================================
    # ЧАСТЬ 2: ПОСТРОЕНИЕ ГРАФИКОВ ЗАВИСИМОСТЕЙ
    # ==========================================
    # Запускается автоматически после закрытия предыдущего графика
    graphs.build_benchmark_graphs()


if __name__ == "__main__":
    run_pipeline()
