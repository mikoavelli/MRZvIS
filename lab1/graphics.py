import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from binary_logic import to_binary, from_binary, sum_binary, inverse_binary

try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans']
except Exception as e:
    print(f"Не удалось установить шрифт для matplotlib: {e}. Кириллица может не отображаться.")


def T1(n, r):
    if n <= 0 or r <= 0: return 0
    return n * r


def Tn(n, r):
    if n <= 0 or r <= 0: return 0
    return n + r - 1


def Ky(n, r):
    t1 = T1(n, r)
    tn = Tn(n, r)
    if tn == 0: return np.nan
    return t1 / tn


def e(n, r):
    ky = Ky(n, r)
    if n == 0: return np.nan
    return ky / n


def plot_ky_vs_r(n_actual, r_actual, digits, r_max_plot=20):
    """
    График Ky от r.
    Точки для n=1 и n=n_actual.
    Для n=n_actual подписываются значения Ky у точек.
    """
    plt.figure(figsize=(8, 6))
    r_values = np.arange(1, r_max_plot + 1)

    ky_n1 = [Ky(1, r_val) for r_val in r_values]
    plt.plot(r_values, ky_n1, marker='o', linestyle='none', label=f'n=1')

    ky_n_actual_vals = [Ky(n_actual, r_val) for r_val in r_values]
    plt.plot(r_values, ky_n_actual_vals, marker='o', linestyle='none', label=f'n={n_actual}')

    for r_val, ky_val in zip(r_values, ky_n_actual_vals):
        if not np.isnan(ky_val):
            plt.text(r_val, ky_val, f'{ky_val:.2f}', fontsize=8, ha='center', va='bottom')
    
    # Добавление горизонтальной линии на уровне y=6
    plt.axhline(y=digits, color='red', linestyle='--', linewidth=1.5, 
                label='$K_y = 6$ асимптота.')

    plt.title(f'График зависимости $K_y$ от ранга задачи $r$')
    plt.xlabel('Ранг задачи, $r$')
    plt.ylabel('Коэффициент ускорения, $K_y$')
    plt.xticks(np.arange(0, r_max_plot + 1, 2))
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend()


def plot_e_vs_r(n_actual, r_actual, r_max_plot=20):
    """
    График e от r.
    Точки для n=1 и n=n_actual.
    Для n=n_actual подписываются значения e у точек.
    """
    plt.figure(figsize=(8, 6))
    r_values = np.arange(1, r_max_plot + 1)

    e_n1 = [e(1, r_val) for r_val in r_values]
    plt.plot(r_values, e_n1, marker='o', linestyle='none', label=f'n=1')

    e_n_actual_vals = [e(n_actual, r_val) for r_val in r_values]
    plt.plot(r_values, e_n_actual_vals, marker='o', linestyle='none', label=f'n={n_actual}')

    for r_val, e_val in zip(r_values, e_n_actual_vals):
        if not np.isnan(e_val):
            plt.text(r_val, e_val, f'{e_val:.2f}', fontsize=8, ha='center', va='bottom',
                     )

    plt.title(f'График зависимости $e$ от ранга задачи $r$')
    plt.xlabel('Ранг задачи, $r$')
    plt.ylabel('Эффективность, $e$')
    plt.xticks(np.arange(0, r_max_plot + 1, 2))
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()


def plot_ky_vs_n(
    n_actual_sim_param, # Параметр из симуляции, если нужен для определения n_max_plot
    r_actual_sim_param, # Параметр из симуляции, если нужен для определения r_max_curves
    n_max_plot_param=None,
    r_max_curves_param=None,
    ky_horizontal_line_at=8 # Значение для горизонтальной линии
):
    """
    График Ky от n.
    Только начальная (n=1) и конечная (n=n_max_plot_param) точки для каждой кривой r.
    Добавлена горизонтальная линия на Ky = ky_horizontal_line_at.
    """
    # Определение n_max_plot и r_max_curves, если не переданы явно
    if n_max_plot_param is None:
        n_max_plot_param = max(n_actual_sim_param + 2, 10) # Например, n_max_plot = 10 как на скриншоте
    if r_max_curves_param is None:
        r_max_curves_param = max(r_actual_sim_param, 8) # Например, r до 9 как на скриншоте

    plt.figure(figsize=(8, 6))

    # Точки n, для которых будут строиться значения Ky
    # Обычно это n=1 и n=n_max_plot_param
    n_points_to_plot = [1]
    if n_max_plot_param > 1: # Добавляем конечную точку, только если она отличается от начальной
        n_points_to_plot.append(n_max_plot_param)
    # Убираем дубликаты и сортируем, если n_max_plot_param=1
    n_points_to_plot = sorted(list(set(n_points_to_plot)))


    r_fixed_values = np.arange(1, r_max_curves_param + 1)

    legend_handles = []
    legend_labels = []
    max_y_value_from_points = 0 # Для автоматического масштабирования оси Y

    for r_fix in r_fixed_values:
        # Вычисляем Ky только для начальной и конечной точек n
        ky_at_endpoints = [Ky(n_val, r_fix) for n_val in n_points_to_plot]

        # Обновляем максимальное значение Y, встреченное на графике
        for ky_val in ky_at_endpoints:
            if not np.isnan(ky_val):
                max_y_value_from_points = max(max_y_value_from_points, ky_val)

        # Рисуем только маркеры, без соединительной линии
        line, = plt.plot(n_points_to_plot, ky_at_endpoints, marker='o', linestyle='none', label=f'r={r_fix}')
        legend_handles.append(line)
        legend_labels.append(f'r={r_fix}')

    # --- ДОБАВЛЕНИЕ ГОРИЗОНТАЛЬНОЙ ЛИНИИ ---
    # if ky_horizontal_line_at is not None:
    #     # Рисуем линию через весь диапазон x, используемый для точек
    #     # Если n_points_to_plot содержит только одну точку, расширяем диапазон для линии
    #     x_line_min = min(n_points_to_plot)
    #     x_line_max = max(n_points_to_plot)
    #     if x_line_min == x_line_max: # Только одна точка n
    #         x_line_coords_for_horizontal = [x_line_min - 0.5, x_line_max + 0.5]
    #     else:
    #         x_line_coords_for_horizontal = [x_line_min, x_line_max]

    #     # Для большей наглядности можно рисовать линию чуть шире, чем крайние точки n
    #     # x_line_coords_for_horizontal = [0.5, n_max_plot_param + 0.5]

    #     line_h, = plt.plot(x_line_coords_for_horizontal, [ky_horizontal_line_at] * len(x_line_coords_for_horizontal),
    #                        color='red', linestyle='--', linewidth=1.5,
    #                        label=f'$K_y = {ky_horizontal_line_at}$ асимптота.')
    #     legend_handles.append(line_h)
    #     legend_labels.append(f'$K_y = {ky_horizontal_line_at}$ - асимптота')
    #     max_y_value_from_points = max(max_y_value_from_points, ky_horizontal_line_at)


    plt.title(f'График зависимости $K_y$ от кол-ва этапов $n$ (начало и конец)')
    plt.xlabel('Кол-во этапов, $n$')
    plt.ylabel('Коэффициент ускорения, $K_y$')

    # Настройка меток на оси X
    # Показываем все целые числа от 1 до n_max_plot_param
    final_xticks = np.arange(1, n_max_plot_param + 1, 1).astype(int)
    # Убедимся, что начальная и конечная точки n (если они целые) включены
    if 1 not in final_xticks and n_max_plot_param >=1 : final_xticks = np.insert(final_xticks, 0, 1)
    if n_max_plot_param not in final_xticks and n_max_plot_param >=1 : final_xticks = np.append(final_xticks, n_max_plot_param)
    final_xticks = np.unique(final_xticks)
    # Показываем только те метки, которые попадают в диапазон [1, n_max_plot_param]
    final_xticks = final_xticks[(final_xticks >= 1) & (final_xticks <= n_max_plot_param)]

    if len(final_xticks) > 0 :
        plt.xticks(final_xticks)
    else: # Если n_max_plot_param < 1, например
        plt.xticks([1])


    # Пределы по осям
    plt.xlim(left=0.5, right=n_max_plot_param + 0.5) # Немного отступа по краям
    plt.ylim(bottom=0, top=max(1, max_y_value_from_points) * 1.1) # От 0 до чуть выше максимального значения

    plt.grid(True)
    plt.legend(legend_handles, legend_labels)
    plt.show()


def plot_e_vs_n(
    n_actual_sim_param,
    r_actual_sim_param,
    n_max_plot_param=None,
    r_max_curves_param=None,
    extra_point_coords=(10, 1.0) # Координаты (n, e) для дополнительной точки
):
    """
    График e от n.
    Только начальная (n=1) и конечная (n=n_max_plot_param) точки для каждой кривой r.
    Добавлена дополнительная выделенная точка.
    """
    if n_max_plot_param is None:
        n_max_plot_param = max(n_actual_sim_param + 2, 10)
    if r_max_curves_param is None:
        r_max_curves_param = max(r_actual_sim_param, 8)

    plt.figure(figsize=(8, 6))
    n_points_to_plot = [1]
    if n_max_plot_param > 1:
        n_points_to_plot.append(n_max_plot_param)
    n_points_to_plot = sorted(list(set(n_points_to_plot)))

    r_fixed_values = np.arange(1, r_max_curves_param + 1)

    legend_handles = []
    legend_labels = []

    for r_fix in r_fixed_values:
        e_at_endpoints = [e(n_val, r_fix) for n_val in n_points_to_plot]
        line, = plt.plot(n_points_to_plot, e_at_endpoints, marker='o', linestyle='none', label=f'r={r_fix}')
        legend_handles.append(line)
        legend_labels.append(f'r={r_fix}')

    # --- ДОБАВЛЕНИЕ ДОПОЛНИТЕЛЬНОЙ ТОЧКИ ---
    # if extra_point_coords:
    #     n_extra, e_extra = extra_point_coords
    #     # Рисуем точку большим маркером другого цвета/формы
    #     extra_pt_handle = plt.scatter([n_extra], [e_extra],
    #                                   color='red', marker='*', s=150, # Красная звезда, размер 150
    #                                   label=f'Ассимтота ({n_extra}, {e_extra:.2f})',
    #                                   zorder=5) # zorder чтобы быть поверх других точек
    #     legend_handles.append(extra_pt_handle)
    #     legend_labels.append(f'Ассимтота ({n_extra}, {e_extra:.2f})')


    plt.title(f'График зависимости $e$ от кол-ва этапов $n$ (начало и конец)')
    plt.xlabel('Кол-во этапов, $n$')
    plt.ylabel('Эффективность, $e$')

    # Настройка меток на оси X
    step_x = 1 if n_max_plot_param <= 15 else int(n_max_plot_param / 10)
    # Включаем основные точки и, если есть, координату n дополнительной точки
    xticks_values = list(np.arange(1, n_max_plot_param + 1, step_x))
    xticks_values.extend(n_points_to_plot)
    if extra_point_coords:
        xticks_values.append(extra_point_coords[0])

    final_xticks = np.unique(np.array(xticks_values).astype(int))
    final_xticks = final_xticks[(final_xticks >= 1) & (final_xticks <= n_max_plot_param + 1)] # +1 для видимости последней точки
    if len(final_xticks) > 0:
        plt.xticks(final_xticks)
    else:
        plt.xticks([1])


    # Пределы по осям
    # X-ось: немного шире, чем n_max_plot_param и координата n доп. точки
    max_n_on_plot = n_max_plot_param
    if extra_point_coords:
        max_n_on_plot = max(max_n_on_plot, extra_point_coords[0])
    plt.xlim(left=0.5, right=max_n_on_plot + 0.5)

    # Y-ось: от 0 до 1.1, или чуть выше если доп. точка выходит за пределы
    max_y_on_plot = 1.1
    if extra_point_coords and extra_point_coords[1] > 1.0:
        max_y_on_plot = max(max_y_on_plot, extra_point_coords[1] * 1.1)
    plt.ylim(0, max_y_on_plot)

    plt.grid(True)
    plt.legend(legend_handles, legend_labels)
    plt.show()


class MyException(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


class Conveyer:
    """Non-restoring division conveyer"""

    def __init__(self,
                 list_dividend: list[str],
                 list_divisor: list[str],
                 num_of_bin_digits: int
                 ) -> None:
        self._dividends, self._divisors = self._check_values(list_dividend, list_divisor, num_of_bin_digits)
        self._count: int = num_of_bin_digits
        self._num_tasks: int = len(self._dividends)
        self._generate_conveyer()
        self._start_conveyer()
        self._generate_and_show_plots()

    @staticmethod
    def _check_values(list_dividend: list[str],
                      list_divisor: list[str],
                      num_of_bin_digits: int
                      ) -> tuple[list[int], list[int]]:
        try:
            dividends: list[int] = [int(num) for num in list_dividend]
            divisors: list[int] = [int(num) for num in list_divisor]

            if num_of_bin_digits <= 0: raise MyException("Count (n) must be > 0")
            if len(dividends) <= 0: raise MyException("Number of tasks (r) must be > 0")
            if len(dividends) != len(divisors): raise MyException("Dividend/divisor list length mismatch")

            max_val = 2 ** num_of_bin_digits - 1
            for i in range(len(dividends)):
                if not (0 <= dividends[i] <= max_val):
                    raise MyException(
                        f"Dividend {dividends[i]} out of range [0, {max_val}] for {num_of_bin_digits} bits")
                if not (0 < divisors[i] <= max_val):
                    raise MyException(f"Divisor {divisors[i]} out of range (0, {max_val}] for {num_of_bin_digits} bits")

        except ValueError:
            raise MyException("Inputs must be integers")
        except MyException as e:
            print(f"Input Error: {e}. Aborting.")
            exit(1)
        return dividends, divisors

    def _generate_conveyer(self) -> None:
        self._conveyer: dict = {'queue': tuple(zip(self._dividends, self._divisors))}
        for i in range(self._count): self._conveyer[f"step{i + 1}"] = None
        self._conveyer['result'] = None
        self._conveyer['queue_backup'] = tuple(zip(self._dividends, self._divisors))
        self._conveyer['tact'] = 0
        self._max_tacts = self._count + self._num_tasks - 1

    def _start_conveyer(self) -> None:
        print("--- Starting Conveyer Simulation ---")
        target_tacts = self._count + self._num_tasks
        while self._conveyer['tact'] < target_tacts:
            self._next()
            self._conveyer['tact'] += 1
        self._conveyer['tact'] = self._max_tacts
        print(f"\n--- Simulation Complete (Total Tacts: {self._max_tacts}) ---")
        self._print_conveyer()
        print("--- End of Simulation Output ---")

    def _print_conveyer(self) -> None:
        print(f"Tact: {self._conveyer['tact']}")
        print("\nEnter queue:", end='')
        if self._conveyer['queue']:
            print(" ".join([f"{d}/{s}" for d, s in self._conveyer['queue']]))
        elif self._conveyer['tact'] >= self._num_tasks:
            print(" (Empty)")
        else:
            print(" (Initial queue processed)")

        print("\nPipeline Stages:")
        for i in range(1, self._count + 1):
            print(f" Step {i}: ", end='')
            step = self._conveyer.get(f"step{i}")
            if not step:
                print("(Empty)")
            else:
                print(f"R={step['reminder']} A={step['dividend']} D={step['divisor']}")

        print("\nResult buffer:")
        if self._conveyer['result'] is not None:
            for idx, result in enumerate(self._conveyer['result']):
                div_orig, dsor_orig = self._conveyer['queue_backup'][idx]
                q_bin = result['dividend']
                r_bin = result['reminder']
                r_bin_int_part = r_bin[:self._count]
                try:
                    q_int = from_binary(q_bin)
                    r_int = from_binary(r_bin_int_part)
                    print(
                        f"  Task {idx + 1} ({div_orig}/{dsor_orig}): Q={q_bin} ({q_int}), R={r_bin_int_part} ({r_int})")
                except ValueError:
                    print(f"  Task {idx + 1} ({div_orig}/{dsor_orig}): Q={q_bin} (...), R={r_bin_int_part} (...)")
        else:
            print("  (No results yet)")
        print('-' * 30)

    def _next(self) -> None:
        step_last = self._conveyer.get(f"step{self._count}")
        if step_last:
            result_item = self._calculate_last_step(step_last)
            if self._conveyer['result'] is None:
                self._conveyer['result'] = (result_item,)
            else:
                self._conveyer['result'] += (result_item,)
            self._conveyer[f"step{self._count}"] = None

        for i in range(self._count - 1, 0, -1):
            step_current = self._conveyer.get(f"step{i}")
            if step_current:
                self._conveyer[f"step{i + 1}"] = self._calculate_step(step_current)
                self._conveyer[f"step{i}"] = None

        if self._conveyer['queue']:
            dividend, divisor = self._conveyer['queue'][0]
            initial_state: dict[str, str] = {
                'reminder': '0' * (self._count + 1),
                'dividend': to_binary(dividend, self._count),
                'divisor': to_binary(divisor, self._count + 1)
            }
            self._conveyer['step1'] = self._calculate_step(initial_state)
            self._conveyer['queue'] = self._conveyer['queue'][1:]

    def _calculate_last_step(self, step: dict[str, str]) -> dict[str, str]:
        final_step = step.copy()
        if final_step['reminder'][0] == '1':
            final_step['reminder'] = sum_binary(final_step['reminder'], final_step['divisor'])
        return final_step

    def _calculate_step(self, step: dict[str, str, str]):
        next_step = step.copy()
        is_sum_operation = next_step['reminder'][0] == '1'
        next_step['reminder'], next_step['dividend'] = self._shift_and_operation(
            next_step['reminder'], next_step['dividend'], next_step['divisor'],
            is_sum_operation=is_sum_operation
        )
        quotient_digit = '0' if next_step['reminder'][0] == '1' else '1'
        next_step['dividend'] = next_step['dividend'][:-1] + quotient_digit
        return next_step

    @staticmethod
    def _shift_and_operation(reminder: str, dividend: str, divisor: str, is_sum_operation: bool) -> tuple[str, str]:
        new_reminder = reminder[1:] + dividend[0]
        new_dividend = dividend[1:] + '_'
        if is_sum_operation:
            operated_reminder = sum_binary(new_reminder, divisor)
        else:
            operated_reminder = sum_binary(new_reminder, inverse_binary(divisor))
        return operated_reminder, new_dividend

    def _generate_and_show_plots(self):
        print("\n--- Generating Performance Graphs ---")
        n_sim = self._count
        r_sim = self._num_tasks

        if n_sim <= 0 or r_sim <= 0:
            print("Cannot generate plots for n=0 or r=0.")
            return

        print(f"Plotting theoretical performance for n={n_sim}, r={r_sim} (without highlighting)")

        r_plot_limit = max(20, r_sim + 5)
        n_plot_limit = max(n_sim + 2, 8)
        r_curves_limit = max(8, r_sim + 1)

        plot_ky_vs_r(n_sim, r_sim, n_sim, r_max_plot=r_plot_limit)
        plot_e_vs_r(n_sim, r_sim, r_max_plot=r_plot_limit)
        plot_ky_vs_n(n_sim, r_sim, n_max_plot_param=n_plot_limit, r_max_curves_param=r_curves_limit)
        plot_e_vs_n(n_sim, r_sim, n_max_plot_param=n_plot_limit, r_max_curves_param=r_curves_limit)

        plt.show()
        print("--- End of Plotting ---")


if __name__ == "__main__":

    try:
        # dividends_in = input("Enter space-separated dividends: ").split()
        # divisors_in = input("Enter space-separated divisors: ").split()
        # n_digits_in = int(input("Enter number of binary digits (pipeline stages, n): "))
        n_digits_in = 8
        rank = 10
        dividends_in = [str(random.randint(0, 2 ** n_digits_in - 1)) for _ in range(rank)]
        divisors_in = [str(random.randint(1, 2 ** n_digits_in - 1)) for _ in range(rank)]

        conveyor_instance = Conveyer(dividends_in, divisors_in, n_digits_in)

    except ValueError:
        print("Error: Number of binary digits must be an integer.")
        exit(1)
    except MyException as e:
        print(f"Error: {e}")
        exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)