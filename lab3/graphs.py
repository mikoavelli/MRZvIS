# Индивидуальная лабораторная работа 3 по дисциплине МРЗвИС вариант 6
# "Реализовать модель линейной рециркуляционной  сети с адаптивным коэффициентом обучения с ненормированными весами
# с логарифмисечкой функциоей активации (гиперболический арксинус) выходного сигнала на скрытом слое"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import concurrent.futures

import matplotlib
import numpy as np

matplotlib.use("Agg")
import os
import time

import matplotlib.pyplot as plt
from data_utils import pad_and_chunk_image

# Импорт ваших классов
from model import ImageReconstructorNN
from PIL import Image

# =================================================================
# КОНФИГУРАЦИЯ
# =================================================================

IMAGE_PATHS = ["input.bmp", "input2.bmp", "input3.bmp"]
PLOT_OUTPUT_DIR = "plots_sequential"

CHUNK_SHAPE = (8, 8)
INPUT_SIZE = 8 * 8 * 3  # 192
GLOBAL_MAX_EPOCHS = 1000
GLOBAL_LOG_FREQUENCY = 20

# Параметры графиков
EXP1_ERRORS = [3000, 2500, 2000, 1500, 1000, 750]
EXP1_HIDDEN = 64
EXP1_LR = 0.005

EXP2_HIDDEN_SIZES = [192, 96, 64, 48, 32]
EXP2_TARGET_ERROR = 750
EXP2_LR = 0.005
EXP2_RUNS = 2

EXP3_LRS = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3]
EXP3_HIDDEN = 64
EXP3_TARGET_ERROR = 750

# Параметры адаптивности (Bold Driver)
LR_PATIENCE_DECREASE = 5
LR_FACTOR_DECREASE = 0.5
LR_PATIENCE_INCREASE = 3
LR_FACTOR_INCREASE = 1.1
LR_MIN_RATE = 1e-6
LR_MAX_RATE = 0.005


# =================================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =================================================================


def task_wrapper(args):
    """Распаковщик аргументов для ProcessPoolExecutor"""
    return run_training_session(*args)


def ensure_dirs_and_images():
    if not os.path.exists(PLOT_OUTPUT_DIR):
        os.makedirs(PLOT_OUTPUT_DIR)

    for i, path in enumerate(IMAGE_PATHS):
        if not os.path.exists(path):
            arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            if i == 1:
                arr = arr // 2
            if i == 2:
                arr[:, :, 0] = 0
            Image.fromarray(arr).save(path)


def run_training_session(image_path, hidden_size, target_loss, initial_lr, tag=""):
    """
    Запускает одну сессию обучения последовательно.
    """
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img) / 255.0
        chunks, _, _ = pad_and_chunk_image(img_arr, CHUNK_SHAPE)

        model = ImageReconstructorNN(INPUT_SIZE, hidden_size, INPUT_SIZE)

        config = {
            "target_loss": target_loss,
            "max_epochs": GLOBAL_MAX_EPOCHS,
            "initial_lr": initial_lr,
            "log_frequency": GLOBAL_LOG_FREQUENCY,
            "batch_size": 1,
            "enable_scheduler": True,
            "lr_patience_decrease": LR_PATIENCE_DECREASE,
            "lr_factor_decrease": LR_FACTOR_DECREASE,
            "lr_patience_increase": LR_PATIENCE_INCREASE,
            "lr_factor_increase": LR_FACTOR_INCREASE,
            "min_lr": LR_MIN_RATE,
        }

        print(f"[{tag}] Start training (Target Loss: {target_loss}, LR: {initial_lr})...")
        epochs = model.train(chunks, chunks, config)
        print(f"[{tag}] Finished in {epochs} epochs.")
        return epochs
    except Exception as e:
        print(f"[{tag}] Error: {e}")
        return GLOBAL_MAX_EPOCHS


# =================================================================
# ФУНКЦИИ ПОСТРОЕНИЯ (Последовательные)
# =================================================================


def plot_1_error():
    """Процесс 1: Параллельный расчет и построение графика ошибки."""
    tag = "PLOT 1"
    print(f"\n[{tag}] Подготовка задач...")

    tasks = []
    # Формируем плоский список задач, сохраняя порядок
    for path in IMAGE_PATHS:
        for err in EXP1_ERRORS:
            # (args...)
            tasks.append((path, EXP1_HIDDEN, err, EXP1_LR, f"{tag} {path}"))

    print(f"[{tag}] Запуск {len(tasks)} задач параллельно...")

    # Считаем параллельно (ВНУТРИ графика используем мультипроцессинг)
    with concurrent.futures.ProcessPoolExecutor() as pool:
        results_flat = list(pool.map(task_wrapper, tasks))

    # Распаковка и рисование
    print(f"[{tag}] Расчет окончен. Построение графика...")

    plt.figure(figsize=(10, 6))

    # Мы знаем порядок задач: сначала все Error для Image1, потом для Image2...
    idx_counter = 0
    for path in IMAGE_PATHS:
        y_epochs = []
        for _ in EXP1_ERRORS:
            ep = results_flat[idx_counter]
            y_epochs.append(ep)
            idx_counter += 1

        plt.plot(EXP1_ERRORS, y_epochs, marker="o", label=path)

    plt.title("Зависимость итераций от допустимой ошибки")
    plt.xlabel("Максимально допустимая ошибка (SSE)")
    plt.ylabel("Количество итераций")
    plt.gca().invert_xaxis()
    plt.grid(True)
    plt.legend()

    out_path = f"{PLOT_OUTPUT_DIR}/1_error.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[{tag}] График сохранен: {out_path}")


def plot_2_compression():
    """Процесс 2: Параллельный расчет и построение графика сжатия."""
    tag = "PLOT 2"
    print(f"\n[{tag}] Подготовка задач...")

    tasks = []
    image_path = IMAGE_PATHS[0]

    # Задачи: для каждого Hidden Size запускаем EXP2_RUNS раз
    for h_size in EXP2_HIDDEN_SIZES:
        for i in range(EXP2_RUNS):
            tasks.append((image_path, h_size, EXP2_TARGET_ERROR, EXP2_LR, f"{tag} H={h_size}"))

    print(f"[{tag}] Запуск {len(tasks)} задач параллельно...")

    with concurrent.futures.ProcessPoolExecutor() as pool:
        results_flat = list(pool.map(task_wrapper, tasks))

    print(f"[{tag}] Расчет окончен. Построение графика...")

    # Усреднение результатов
    avg_epochs = []
    ratios = [round(INPUT_SIZE / h, 2) for h in EXP2_HIDDEN_SIZES]

    # Разбиваем плоский список на куски по EXP2_RUNS
    for i in range(0, len(results_flat), EXP2_RUNS):
        chunk = results_flat[i : i + EXP2_RUNS]
        avg = sum(chunk) / len(chunk)
        avg_epochs.append(avg)

    plt.figure(figsize=(10, 6))
    plt.plot(ratios, avg_epochs, marker="s", color="orange")
    plt.title("Зависимость итераций от коэффициента сжатия")
    plt.xlabel("Коэффициент сжатия")
    plt.ylabel("Среднее количество итераций")
    plt.grid(True)

    for x, y in zip(ratios, avg_epochs):
        plt.text(x, y, f"{y:.0f}", ha="center", va="bottom")

    out_path = f"{PLOT_OUTPUT_DIR}/2_compression.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[{tag}] График сохранен: {out_path}")


def plot_3_lr():
    """Процесс 3: Параллельный расчет и построение графика LR."""
    tag = "PLOT 3"
    print(f"\n[{tag}] Подготовка задач...")

    tasks = []
    image_path = IMAGE_PATHS[0]

    for lr in EXP3_LRS:
        tasks.append((image_path, EXP3_HIDDEN, EXP3_TARGET_ERROR, lr, f"{tag} LR={lr}"))

    print(f"[{tag}] Запуск {len(tasks)} задач параллельно...")

    with concurrent.futures.ProcessPoolExecutor() as pool:
        results = list(pool.map(task_wrapper, tasks))

    print(f"[{tag}] Расчет окончен. Построение графика...")

    plt.figure(figsize=(10, 6))
    plt.plot(EXP3_LRS, results, marker="^", color="green")
    plt.title("Зависимость итераций от начального LR")
    plt.xlabel("Learning Rate")
    plt.ylabel("Количество итераций")
    plt.xscale("log")
    plt.grid(True, which="both")

    out_path = f"{PLOT_OUTPUT_DIR}/3_lr.png"
    plt.savefig(out_path)
    plt.close()
    print(f"[{tag}] График сохранен: {out_path}")


# =================================================================
# MAIN
# =================================================================

if __name__ == "__main__":
    ensure_dirs_and_images()

    start_time = time.time()
    print("Запуск последовательной генерации графиков.")

    # plot_1_error()
    plot_2_compression()
    # plot_3_lr()

    end_time = time.time()
    print("\n==================================================")
    print(f"Все графики построены за: {end_time - start_time:.2f} сек.")
    print(f"Результаты в папке: {PLOT_OUTPUT_DIR}")
