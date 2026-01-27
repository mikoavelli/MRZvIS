# Индивидуальная лабораторная работа 2 по дисциплине МРЗвИС вариант 6
# "Реализовать модель сети Хэмминга"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import matplotlib.pyplot as plt
import numpy as np
import utils
from hamming_network import HammingNetwork


def run_noise_experiment(patterns, labels, steps=20, trials_per_step=10):
    """
    Исследует зависимость поведения сети от уровня шума.
    Строит два графика:
    1. Точность распознавания от уровня шума.
    2. Среднее количество итераций релаксации от уровня шума.
    """
    noise_levels = np.linspace(0, 1.0, steps)  # От 0% до 100% шума
    avg_iterations = []
    accuracies = []

    net = HammingNetwork()
    net.fit(patterns, labels)

    print(f"Запуск эксперимента по шуму ({steps} шагов)...")

    for noise in noise_levels:
        correct_count = 0
        iters_sum = 0

        # Прогоняем N раз для усреднения
        total_trials = len(patterns) * trials_per_step

        for _ in range(trials_per_step):
            for i, original_vec in enumerate(patterns):
                noisy_vec = utils.add_noise(original_vec, noise)
                pred_label, iters, _ = net.predict(noisy_vec)

                if pred_label == labels[i]:
                    correct_count += 1
                iters_sum += iters

        accuracies.append((correct_count / total_trials) * 100)
        avg_iterations.append(iters_sum / total_trials)

    # --- Визуализация ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # График 1: Точность
    ax1.plot(noise_levels * 100, accuracies, marker="o", color="green", label="Точность")
    ax1.set_title("Устойчивость к шуму")
    ax1.set_xlabel("Уровень шума (%)")
    ax1.set_ylabel("Точность распознавания (%)")
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.axvline(x=50, color="red", linestyle="--", label="50% шума (Граница)")
    ax1.legend()

    # График 2: Итерации
    ax2.plot(
        noise_levels * 100,
        avg_iterations,
        marker="s",
        color="blue",
        label="Итерации MAXNET",
    )
    ax2.set_title("Скорость сходимости (Релаксация)")
    ax2.set_xlabel("Уровень шума (%)")
    ax2.set_ylabel("Среднее кол-во итераций")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.suptitle("Характеристики сети Хэмминга при зашумлении", fontsize=16)
    plt.show()


def run_capacity_experiment(patterns, labels):
    """
    Исследует, как количество запомненных образов влияет на время релаксации.
    Сеть Хэмминга: чем больше образов, тем больше нейронов в MAXNET, тем дольше идет конкуренция.
    """
    max_patterns = len(patterns)
    if max_patterns < 2:
        print("Недостаточно данных для теста емкости.")
        return

    subset_sizes = range(2, max_patterns + 1)
    avg_iters_by_size = []

    print(f"Запуск эксперимента по емкости (от 2 до {max_patterns} классов)...")

    for size in subset_sizes:
        # Берем подмножество данных
        sub_patterns = patterns[:size]
        sub_labels = labels[:size]

        net = HammingNetwork()
        net.fit(sub_patterns, sub_labels)

        iters_sum = 0
        trials = 50  # Много прогонов для точности

        # Тестируем на случайных образах из этого подмножества с фиксированным шумом
        fixed_noise = 0.2

        for _ in range(trials):
            idx = np.random.randint(0, size)
            vec = sub_patterns[idx]
            noisy_vec = utils.add_noise(vec, fixed_noise)
            _, iters, _ = net.predict(noisy_vec)
            iters_sum += iters

        avg_iters_by_size.append(iters_sum / trials)

    # --- Визуализация ---
    plt.figure(figsize=(10, 6))
    plt.plot(subset_sizes, avg_iters_by_size, marker="D", color="purple")
    plt.title("Зависимость итераций релаксации от количества классов")
    plt.xlabel("Количество запомненных образов (M)")
    plt.ylabel("Итерации MAXNET")
    plt.grid(True)
    plt.show()


def run_resolution_experiment(original_file_paths):
    """
    Исследует зависимость от размера изображения (N).
    Загружает одни и те же картинки, ресайзит их в разные размеры (10x10, 20x20...)
    и замеряет итерации.
    """
    resolutions = [10, 20, 30, 40, 50, 60]
    avg_iters_by_res = []

    print("Запуск эксперимента по разрешению...")

    for res in resolutions:
        size = (res, res)
        # Перезагружаем данные с новым размером
        patterns = []
        labels = []
        for path in original_file_paths:
            vec, _ = utils.load_image_as_vector(path, target_size=size)
            if vec is not None:
                patterns.append(vec)
                labels.append("dummy")  # Метки не важны для замера скорости

        net = HammingNetwork()
        net.fit(patterns, labels)

        iters_sum = 0
        trials = 50
        fixed_noise = 0.2

        for _ in range(trials):
            idx = np.random.randint(0, len(patterns))
            vec = patterns[idx]
            noisy_vec = utils.add_noise(vec, fixed_noise)
            _, iters, _ = net.predict(noisy_vec)
            iters_sum += iters

        avg_iters_by_res.append(iters_sum / trials)

    # --- Визуализация ---
    plt.figure(figsize=(10, 6))
    plt.plot(resolutions, avg_iters_by_res, marker="o", color="orange")
    plt.title("Зависимость итераций релаксации от размерности входа (N)")
    plt.xlabel("Размер стороны изображения (px)")
    plt.ylabel("Итерации MAXNET")
    plt.grid(True)

    # Теоретически для Хэмминга этот график должен быть почти плоским,
    # так как MAXNET зависит от M (кол-во классов), а не от N (пикселей).
    # N влияет только на расчет первого слоя (очень быстро).
    plt.show()
