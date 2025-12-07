import os

import config
import matplotlib.pyplot as plt
import utils
from hamming_network import HammingNetwork


def load_dataset():
    """
    Загружает изображения.
    Автоматически определяет размер по ПЕРВОМУ найденному изображению.
    Остальные приводит к этому размеру.
    """
    patterns = []
    labels = []

    # Генерация если пусто
    utils.generate_dummy_data()

    files = sorted(
        [
            f
            for f in os.listdir(config.DATA_DIR)
            if f.lower().endswith((".png", ".jpg", ".bmp", ".jpeg"))
        ]
    )

    if not files:
        raise FileNotFoundError(f"Нет изображений в {config.DATA_DIR}")

    # --- Шаг 1: Определяем базовый размер по первому файлу ---
    first_path = os.path.join(config.DATA_DIR, files[0])
    _, detected_shape = utils.load_image_as_vector(first_path, target_size=None)

    if detected_shape is None:
        raise ValueError("Не удалось прочитать первое изображение.")

    print(f"Обнаружен размер изображений: {detected_shape} (WxH)")

    # --- Шаг 2: Загружаем все, приводя к detected_shape ---
    print(f"Загрузка {len(files)} файлов...")

    for f in files:
        path = os.path.join(config.DATA_DIR, f)
        # Передаем detected_shape, чтобы гарантировать одинаковый размер векторов
        vec, shape = utils.load_image_as_vector(path, target_size=detected_shape)

        if vec is not None:
            patterns.append(vec)
            labels.append(os.path.splitext(f)[0])

    return patterns, labels, detected_shape


def visualize_results(results_list, patterns, shape_hw):
    """
    shape_hw: кортеж (width, height) для восстановления картинки
    """
    n_samples = len(results_list)
    # Динамическая высота окна
    fig, axes = plt.subplots(nrows=n_samples, ncols=3, figsize=(10, 3 * n_samples))

    if n_samples == 1:
        axes = [axes]  # wrap in list

    fig.suptitle(
        f"Сеть Хэмминга (Размер: {shape_hw}, Шум: {int(config.NOISE_LEVEL * 100)}%)",
        fontsize=16,
    )

    for i, (noisy_vec, pred_lbl, true_lbl, iters, orig_idx) in enumerate(results_list):
        ax_row = axes[i]

        # 1. Оригинал
        # patterns[orig_idx] берем из памяти
        original_img = utils.vector_to_matrix(patterns[orig_idx], shape_hw)
        ax_row[0].imshow(original_img, cmap="gray", vmin=0, vmax=255)
        ax_row[0].set_title(f"Эталон: {true_lbl}")
        ax_row[0].axis("off")

        # 2. Вход с шумом
        noisy_img = utils.vector_to_matrix(noisy_vec, shape_hw)
        ax_row[1].imshow(noisy_img, cmap="gray", vmin=0, vmax=255)
        ax_row[1].set_title("Вход (Шум)")
        ax_row[1].axis("off")

        # 3. Результат
        status = "OK" if pred_lbl == true_lbl else "ERROR"
        color = "green" if status == "OK" else "red"

        # Текст
        info_text = f"Рез: {pred_lbl}\nИтер: {iters}\n{status}"

        # Для визуализации берем картинку победившего эталона (если нашли такой класс)
        try:
            # Ищем вектор эталона, соответствующий предсказанной метке.
            # (Предполагаем, что labels уникальны, иначе берем первый попавшийся)
            # В данном скрипте labels идут синхронно с patterns, но для безопасности ищем заново:
            # Нам нужно найти индекс предсказанной метки в списке меток, который был в load_dataset.
            # Но здесь labels недоступны, только predicted_label.
            # Упрощение: мы просто выведем текст, а картинку покажем "оригинала" (patterns[orig_idx])
            # с пометкой, совпала она или нет.

            # Но красивее показать, что сеть "увидела".
            # Передадим победивший индекс в results_list (добавим его в main)
            pass
        except:
            pass

        ax_row[2].text(
            0.5,
            0.5,
            info_text,
            ha="center",
            va="center",
            fontsize=12,
            color=color,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor=color),
        )
        ax_row[2].set_title("Выход сети")
        ax_row[2].axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


def main():
    try:
        # Получаем данные и их геометрию
        patterns, labels, detected_shape = load_dataset()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        return

    # Обучение
    net = HammingNetwork()
    net.fit(patterns, labels)

    print(f"Обучено классов: {len(patterns)}. Входной вектор N={net.input_size}.")
    print("-" * 60)

    results = []
    correct_count = 0

    for i, original_vec in enumerate(patterns):
        true_label = labels[i]

        # Шум
        noisy_vec = utils.add_noise(original_vec, config.NOISE_LEVEL)

        # Прогноз
        pred_label, iters, winner_idx = net.predict(noisy_vec)

        if pred_label == true_label:
            correct_count += 1

        results.append((noisy_vec, pred_label, true_label, iters, i))

        print(f"File: {true_label:<10} -> Pred: {pred_label:<10} (Iter: {iters})")

    acc = (correct_count / len(patterns)) * 100
    print("-" * 60)
    print(f"Точность: {acc:.2f}%")

    # Визуализация (передаем detected_shape, чтобы знать как рисовать матрицу)
    visualize_results(results, patterns, detected_shape)


if __name__ == "__main__":
    main()
