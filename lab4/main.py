# Индивидуальная лабораторная работа 2 по дисциплине МРЗвИС вариант 6
# "Реализовать модель сети Хэмминга"
# Выполнена студентом группы X БГУИР X X X
# Использованные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import os

import config
import graphs
import matplotlib.pyplot as plt
import utils
from hamming_network import HammingNetwork


def load_dataset_paths():
    """Возвращает список путей к файлам изображений."""
    utils.generate_dummy_data()
    files = sorted([f for f in os.listdir(config.DATA_DIR) if f.lower().endswith((".png", ".jpg", ".bmp", ".jpeg"))])
    paths = [os.path.join(config.DATA_DIR, f) for f in files]
    return paths


def load_dataset_vectors(paths, target_size=None):
    """
    Динамическое определение размера.
    Программа не знает заранее размер картинок. Она берет размер
    первого файла и заставляет все остальные под него подстроиться.
    """
    patterns = []
    labels = []

    # Определяем размер по первому, если не задан
    if target_size is None:
        if not paths:
            return [], [], None
        _, detected_shape = utils.load_image_as_vector(paths[0])
    else:
        detected_shape = target_size

    for path in paths:
        vec, _ = utils.load_image_as_vector(path, target_size=detected_shape)
        if vec is not None:
            patterns.append(vec)
            # Извлекаем имя файла (букву) как метку класса.
            labels.append(os.path.splitext(os.path.basename(path))[0])

    return patterns, labels, detected_shape


def demo_recognition(patterns, labels, detected_shape):
    """Старый режим: демонстрация на картинках."""
    net = HammingNetwork()
    net.fit(patterns, labels)
    print(f"Обучено классов: {len(patterns)}.")

    results = []
    correct_count = 0

    print("-" * 40)
    for i, original_vec in enumerate(patterns):
        true_label = labels[i]
        noisy_vec = utils.add_noise(original_vec, config.NOISE_LEVEL)
        pred_label, iters, _ = net.predict(noisy_vec)

        if pred_label == true_label:
            correct_count += 1
        results.append((noisy_vec, pred_label, true_label, iters, i))
        print(f"Образ: {true_label:<5} | Предсказание: {pred_label:<5} | Итер: {iters}")

    acc = (correct_count / len(patterns)) * 100
    print("-" * 40)
    print(f"Точность: {acc:.2f}%")

    # Визуализация (функцию можно оставить внутри main или вынести в utils)
    visualize_results(results, patterns, detected_shape)


def visualize_results(results_list, patterns, shape_hw):
    """Функция отрисовки таблицы картинок"""
    n_samples = len(results_list)
    fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=(8, 2.5 * n_samples))
    if n_samples == 1:
        axes = [axes]

    for i, (noisy_vec, pred_lbl, true_lbl, iters, orig_idx) in enumerate(results_list):
        ax_row = axes[i]
        # Оригинал
        ax_row[0].imshow(
            utils.vector_to_matrix(patterns[orig_idx], shape_hw),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        ax_row[0].set_title(f"Эталон: {true_lbl}")
        ax_row[0].axis("off")
        # Шум
        ax_row[1].imshow(utils.vector_to_matrix(noisy_vec, shape_hw), cmap="gray", vmin=0, vmax=255)
        ax_row[1].set_title("Вход")
        ax_row[1].axis("off")
        # Результат
        color = "green" if pred_lbl == true_lbl else "red"
        ax_row[2].text(
            0.5,
            0.5,
            f"Рез: {pred_lbl}\nИтер: {iters}",
            ha="center",
            fontsize=12,
            color=color,
            bbox=dict(facecolor="white", edgecolor=color),
        )
        ax_row[2].set_title("Выход")
        ax_row[2].axis("off")
    plt.tight_layout()
    plt.show()


def main():
    """Реализовано переключение режимов: визуальный тест или научная аналитика."""
    paths = load_dataset_paths()
    if not paths:
        print("Нет данных.")
        return

    print("Выберите режим:")
    print("1. Демонстрация распознавания (таблица картинок)")
    print("2. Построение аналитических графиков (как в отчёте)")

    choice = input("Ваш выбор (1/2): ")

    if choice == "1":
        patterns, labels, shape = load_dataset_vectors(paths)
        demo_recognition(patterns, labels, shape)
    elif choice == "2":
        # Загружаем данные
        patterns, labels, _ = load_dataset_vectors(paths)  # Размер берем оригинальный

        # 1. График Шум vs Итерации/Точность
        graphs.run_noise_experiment(patterns, labels)

        # 2. График Количество образов vs Итерации
        graphs.run_capacity_experiment(patterns, labels)

        # 3. График Размер изображения vs Итерации
        graphs.run_resolution_experiment(paths)

    else:
        print("Неверный ввод.")


if __name__ == "__main__":
    main()
