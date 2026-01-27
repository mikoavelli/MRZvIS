# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель рекурентной сети с цепью нейросетевых моделей долгой кратковременной памяти
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import matplotlib.pyplot as plt
import numpy as np
from rnn import RNN
from utils import DataProcessor, TrigonometryStrategy


def plot_prediction(history, future_real, preds_real, strategy_name):
    """
    Строит график: История + Эталон + Прогноз.
    Принимает ОРИГИНАЛЬНЫЕ значения (не логарифмы).
    """
    plt.figure(figsize=(10, 6))

    seq_len = len(history)
    future_steps = len(future_real)

    # История
    plt.plot(range(seq_len), history, "b--o", label="История", markersize=4)

    # Будущее (x координаты)
    x_future = range(seq_len, seq_len + future_steps)

    # Эталон
    plt.plot(x_future, future_real, "g--x", label="Эталон (Future)", linewidth=2)

    # Прогноз
    plt.plot(x_future, preds_real, "r--o", label="Прогноз RNN", linewidth=2)

    plt.title(f"Результат прогнозирования: {strategy_name}")
    plt.xlabel("Шаг времени")
    plt.ylabel("Значение")
    plt.legend()
    plt.grid(True, alpha=0.3)

    print("Отображение графика прогноза... Закройте окно, чтобы продолжить.")
    plt.show()


# === ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ГРАФИКОВ ОТЧЕТА ===


def _run_experiment_for_graph(target_mse, lr, window_size, hidden_size, max_epochs=3000):
    """
    Обучает модель до достижения ошибки или лимита эпох.
    Использует СИНУС (TrigonometryStrategy) как стабильную базу для бенчмарков.
    """
    strat = TrigonometryStrategy()
    hist, _ = strat.generate(50, 5)

    dp = DataProcessor(window_size)
    norm = dp.normalize(hist)
    X, Y = dp.create_windows(norm)

    model = RNN(1, hidden_size, 1)

    for epoch in range(1, max_epochs + 1):
        total_loss = 0
        for i in range(len(X)):
            y_pred, _, h_all = model.forward(X[i], np.zeros((hidden_size, 1)))
            loss = (y_pred - Y[i].reshape(1, 1)) ** 2
            total_loss += loss.item()
            model.backward(X[i], h_all, h_all[-1], y_pred, Y[i].reshape(1, 1), lr)

        if (total_loss / len(X)) <= target_mse:
            return epoch
    return max_epochs


def _calc_mape(hidden_size):
    """Считает MAPE для заданного кол-ва нейронов (на Синусе)"""
    strat = TrigonometryStrategy()
    hist, fut = strat.generate(50, 1)

    dp = DataProcessor(5)
    norm = dp.normalize(hist)
    X, Y = dp.create_windows(norm)

    model = RNN(1, hidden_size, 1)

    # Фиксируем 200 эпох для проверки "емкости" сети
    for _ in range(200):
        for i in range(len(X)):
            yp, _, ha = model.forward(X[i], np.zeros((hidden_size, 1)))
            model.backward(X[i], ha, ha[-1], yp, Y[i].reshape(1, 1), 0.01)

    # Прогноз
    last_win = norm[-5:].reshape(5, 1)
    y_out, _, _ = model.forward(last_win, np.zeros((hidden_size, 1)))
    pred = dp.denormalize(y_out.item())

    actual = fut[0]

    return np.abs((actual - pred) / actual) * 100


def build_benchmark_graphs():
    """Строит 4 графика ПО ОЧЕРЕДИ"""
    print("\n" + "=" * 50)
    print("ГЕНЕРАЦИЯ ГРАФИКОВ ЗАВИСИМОСТЕЙ (БЕНЧМАРКИ)")
    print("Графики будут появляться по очереди.")
    print("ЗАКРОЙТЕ текущее окно с графиком, чтобы начался расчет следующего.")
    print("=" * 50)

    # --- График 1 ---
    print("\n[1/4] Расчет: Итерации vs MSE...")
    mses = [0.02, 0.01, 0.005, 0.001]
    iters_1 = [_run_experiment_for_graph(m, 0.01, 5, 16) for m in mses]

    plt.figure(figsize=(8, 6))
    plt.plot(mses, iters_1, "o-", color="tab:blue", linewidth=2)
    plt.gca().invert_xaxis()  # 0 справа
    plt.title("Зависимость итераций от допустимой ошибки")
    plt.xlabel("Максимально допустимая ошибка (MSE)")
    plt.ylabel("Количество итераций (эпох)")
    plt.grid(True, linestyle="--", alpha=0.7)
    print(">> Отображение графика 1. Закройте окно для продолжения.")
    plt.show()

    # --- График 2 ---
    print("\n[2/4] Расчет: Итерации vs Learning Rate...")
    lrs = [0.001, 0.005, 0.01, 0.05, 0.1]
    iters_2 = [_run_experiment_for_graph(0.005, lr, 5, 16) for lr in lrs]

    plt.figure(figsize=(8, 6))
    plt.plot(lrs, iters_2, "s-", color="tab:green", linewidth=2)
    plt.title("Зависимость итераций от коэффициента обучения")
    plt.xlabel("Коэффициент обучения (Learning Rate)")
    plt.ylabel("Количество итераций (эпох)")
    plt.grid(True, linestyle="--", alpha=0.7)
    print(">> Отображение графика 2. Закройте окно для продолжения.")
    plt.show()

    # --- График 3 ---
    print("\n[3/4] Расчет: Итерации vs Размер окна...")
    wins = [3, 5, 8, 10]
    iters_3 = [_run_experiment_for_graph(0.005, 0.01, w, 16) for w in wins]

    plt.figure(figsize=(8, 6))
    plt.plot(wins, iters_3, "^-", color="tab:red", linewidth=2)
    plt.title("Зависимость итераций от размера скользящего окна")
    plt.xlabel("Размер окна")
    plt.ylabel("Количество итераций (эпох)")
    plt.grid(True, linestyle="--", alpha=0.7)
    print(">> Отображение графика 3. Закройте окно для продолжения.")
    plt.show()

    # --- График 4 ---
    print("\n[4/4] Расчет: MAPE vs Нейроны...")
    neurons = [2, 4, 8, 16, 32]
    mapes = [_calc_mape(n) for n in neurons]

    plt.figure(figsize=(8, 6))
    plt.plot(neurons, mapes, "D-", color="tab:purple", linewidth=2)
    plt.title("Зависимость ошибки (MAPE) от количества нейронов")
    plt.xlabel("Количество нейронов скрытого слоя")
    plt.ylabel("Средняя абсолютная ошибка (%)")
    plt.grid(True, linestyle="--", alpha=0.7)
    print(">> Отображение графика 4. Это последний график.")
    plt.show()

    print("\nВсе графики построены.")
