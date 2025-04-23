import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
# Убедитесь, что у вас есть файл binary_logic.py или замените импорты
# на фактические реализации, если они в этом же файле.
from binary_logic import to_binary, from_binary, sum_binary, inverse_binary

# --- Настройка Matplotlib для кириллицы ---
try:
    mpl.rcParams['font.family'] = 'sans-serif'
    mpl.rcParams['font.sans-serif'] = ['DejaVu Sans'] # Или другой подходящий шрифт
except Exception as e:
    print(f"Не удалось установить шрифт для matplotlib: {e}. Кириллица может не отображаться.")

# --- Теоретические формулы ---
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

# --- Функции для построения графиков (БЕЗ ВЫДЕЛЕНИЯ ТОЧКИ ЗАПУСКА) ---

def plot_ky_vs_r(n_actual, r_actual, r_max_plot=20):
    """Строит график Ky от r (аналог Рис. 7)"""
    plt.figure(figsize=(8, 6))
    r_values = np.arange(1, r_max_plot + 1)

    ky_n1 = [Ky(1, r_val) for r_val in r_values]
    plt.plot(r_values, ky_n1, marker='.', linestyle='-', label=f'n=1 (теор.)')

    ky_n_actual_vals = [Ky(n_actual, r_val) for r_val in r_values]
    plt.plot(r_values, ky_n_actual_vals, marker='.', linestyle='-', label=f'n={n_actual} (теор.)')

    # --- УДАЛЕНО ВЫДЕЛЕНИЕ ТОЧКИ ---
    # ky_current = Ky(n_actual, r_actual)
    # if r_actual <= r_max_plot:
    #     plt.scatter([r_actual], [ky_current], color='red', s=100, zorder=5,
    #                 label=f'Точка запуска (n={n_actual}, r={r_actual})')

    plt.title(f'График зависимости $K_y$ от ранга задачи $r$')
    plt.xlabel('Ранг задачи, $r$')
    plt.ylabel('Коэффициент ускорения, $K_y$')
    plt.xticks(np.arange(0, r_max_plot + 1, 2))
    plt.ylim(bottom=0)
    plt.grid(True)
    plt.legend() # Легенда теперь содержит только линии n=1 и n=n_actual

def plot_e_vs_r(n_actual, r_actual, r_max_plot=20):
    """Строит график e от r (аналог Рис. 8)"""
    plt.figure(figsize=(8, 6))
    r_values = np.arange(1, r_max_plot + 1)

    e_n1 = [e(1, r_val) for r_val in r_values]
    plt.plot(r_values, e_n1, marker='.', linestyle='-', label=f'n=1 (теор.)')

    e_n_actual_vals = [e(n_actual, r_val) for r_val in r_values]
    plt.plot(r_values, e_n_actual_vals, marker='.', linestyle='-', label=f'n={n_actual} (теор.)')

    # --- УДАЛЕНО ВЫДЕЛЕНИЕ ТОЧКИ ---
    # e_current = e(n_actual, r_actual)
    # if r_actual <= r_max_plot:
    #     plt.scatter([r_actual], [e_current], color='red', s=100, zorder=5,
    #                 label=f'Точка запуска (n={n_actual}, r={r_actual})')

    plt.title(f'График зависимости $e$ от ранга задачи $r$')
    plt.xlabel('Ранг задачи, $r$')
    plt.ylabel('Эффективность, $e$')
    plt.xticks(np.arange(0, r_max_plot + 1, 2))
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend()

def plot_ky_vs_n(n_actual, r_actual, n_max_plot=None, r_max_curves=8):
    """Строит ГЛАДКИЙ график Ky от n (аналог Рис. 9) с маркерами на концах."""
    if n_max_plot is None:
        n_max_plot = max(n_actual + 2, 10)
    if r_max_curves is None:
         r_max_curves = max(r_actual, 8)

    plt.figure(figsize=(8, 6))
    n_smooth = np.linspace(1, n_max_plot, 200)
    r_fixed_values = np.arange(1, r_max_curves + 1)

    # Собираем элементы для легенды
    legend_handles = []
    legend_labels = []

    for r_fix in r_fixed_values:
        ky_smooth_values = [Ky(n_val, r_fix) for n_val in n_smooth]
        # Строим гладкую кривую и сохраняем ее для легенды
        line, = plt.plot(n_smooth, ky_smooth_values, linestyle='-', label=f'r={r_fix}') # Упростили label
        legend_handles.append(line)
        legend_labels.append(f'r={r_fix}')

        n_endpoints = [1, n_max_plot]
        ky_endpoints = [Ky(n_val, r_fix) for n_val in n_endpoints]
        # Ставим маркеры в начальной и конечной точках (без отдельной метки в легенде)
        plt.plot(n_endpoints, ky_endpoints, marker='o', linestyle='none', color=line.get_color())

        # --- УДАЛЕНО ВЫДЕЛЕНИЕ ТОЧКИ ---
        # if r_fix == r_actual:
        #     ky_current = Ky(n_actual, r_actual)
        #     plt.scatter([n_actual], [ky_current], color=line.get_color(), edgecolor='red',
        #                 s=150, zorder=5, linewidth=2,
        #                 label=f'Точка запуска (n={n_actual}, r={r_actual})') # Эта метка больше не создается

    plt.title(f'График зависимости $K_y$ от кол-ва этапов $n$ (гладкий)')
    plt.xlabel('Кол-во этапов, $n$')
    plt.ylabel('Коэффициент ускорения, $K_y$')
    step = 1 if n_max_plot <= 15 else int(n_max_plot / 10)
    plt.xticks(np.arange(1, n_max_plot + 1, step))
    plt.xlim(left=0.8, right=n_max_plot + 0.2)
    plt.ylim(bottom=0)
    plt.grid(True)
    # Используем собранные элементы для легенды
    plt.legend(legend_handles, legend_labels)


def plot_e_vs_n(n_actual, r_actual, n_max_plot=None, r_max_curves=8):
    """Строит ГЛАДКИЙ график e от n (аналог Рис. 10) с маркерами на концах."""
    if n_max_plot is None:
        n_max_plot = max(n_actual + 2, 8)
    if r_max_curves is None:
         r_max_curves = max(r_actual, 8)

    plt.figure(figsize=(8, 6))
    n_smooth = np.linspace(1, n_max_plot, 200)
    r_fixed_values = np.arange(1, r_max_curves + 1)

    legend_handles = []
    legend_labels = []

    for r_fix in r_fixed_values:
        e_smooth_values = [e(n_val, r_fix) for n_val in n_smooth]
        line, = plt.plot(n_smooth, e_smooth_values, linestyle='-', label=f'r={r_fix}')
        legend_handles.append(line)
        legend_labels.append(f'r={r_fix}')

        n_endpoints = [1, n_max_plot]
        e_endpoints = [e(n_val, r_fix) for n_val in n_endpoints]
        plt.plot(n_endpoints, e_endpoints, marker='o', linestyle='none', color=line.get_color())

        # --- УДАЛЕНО ВЫДЕЛЕНИЕ ТОЧКИ ---
        # if r_fix == r_actual:
        #     e_current = e(n_actual, r_actual)
        #     plt.scatter([n_actual], [e_current], color=line.get_color(), edgecolor='red',
        #                 s=150, zorder=5, linewidth=2,
        #                 label=f'Точка запуска (n={n_actual}, r={r_actual})')

    plt.title(f'График зависимости $e$ от кол-ва этапов $n$ (гладкий)')
    plt.xlabel('Кол-во этапов, $n$')
    plt.ylabel('Эффективность, $e$')
    step = 1 if n_max_plot <= 15 else int(n_max_plot / 10)
    plt.xticks(np.arange(1, n_max_plot + 1, step))
    plt.xlim(left=0.8, right=n_max_plot + 0.2)
    plt.ylim(0, 1.1)
    plt.grid(True)
    plt.legend(legend_handles, legend_labels)


# --- Класс MyException и Conveyer (без изменений в логике конвейера) ---
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
        # Проверка и инициализация
        self._dividends, self._divisors = self._check_values(list_dividend, list_divisor, num_of_bin_digits)
        self._count: int = num_of_bin_digits # n
        self._num_tasks: int = len(self._dividends) # r
        # Генерация структуры конвейера
        self._generate_conveyer()
        # Запуск симуляции
        self._start_conveyer()
        # Построение графиков после симуляции
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
                    raise MyException(f"Dividend {dividends[i]} out of range [0, {max_val}] for {num_of_bin_digits} bits")
                if not (0 < divisors[i] <= max_val):
                     raise MyException(f"Divisor {divisors[i]} out of range (0, {max_val}] for {num_of_bin_digits} bits")

        except ValueError: raise MyException("Inputs must be integers")
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
        self._conveyer['tact'] = self._max_tacts # Установить финальный такт
        print(f"\n--- Simulation Complete (Total Tacts: {self._max_tacts}) ---")
        self._print_conveyer()
        print("--- End of Simulation Output ---")

    def _print_conveyer(self) -> None:
        print(f"Tact: {self._conveyer['tact']}")
        print("\nEnter queue:", end='')
        if self._conveyer['queue']:
             print(" ".join([f"{d}/{s}" for d,s in self._conveyer['queue']]))
        elif self._conveyer['tact'] >= self._num_tasks : print(" (Empty)")
        else: print(" (Initial queue processed)")

        print("\nPipeline Stages:")
        for i in range(1, self._count + 1):
            print(f" Step {i}: ", end='')
            step = self._conveyer.get(f"step{i}")
            if not step: print("(Empty)")
            else: print(f"R={step['reminder']} A={step['dividend']} D={step['divisor']}")

        print("\nResult buffer:")
        if self._conveyer['result'] is not None:
            for idx, result in enumerate(self._conveyer['result']):
                div_orig, dsor_orig = self._conveyer['queue_backup'][idx]
                q_bin = result['dividend']
                r_bin = result['reminder']
                r_bin_int_part = r_bin[:self._count] # Остаток имеет n значащих бит
                try:
                    q_int = from_binary(q_bin)
                    r_int = from_binary(r_bin_int_part)
                    print(f"  Task {idx+1} ({div_orig}/{dsor_orig}): Q={q_bin} ({q_int}), R={r_bin_int_part} ({r_int})") # Показываем значащую часть остатка
                except ValueError: # На случай если '_' остался (не должно быть в финале)
                     print(f"  Task {idx+1} ({div_orig}/{dsor_orig}): Q={q_bin} (...), R={r_bin_int_part} (...)")
        else:
            print("  (No results yet)")
        print('-' * 30)

    def _next(self) -> None:
        step_last = self._conveyer.get(f"step{self._count}")
        if step_last:
            result_item = self._calculate_last_step(step_last)
            if self._conveyer['result'] is None: self._conveyer['result'] = (result_item,)
            else: self._conveyer['result'] += (result_item,)
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
        if final_step['reminder'][0] == '1': # Коррекция остатка
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
        if is_sum_operation: operated_reminder = sum_binary(new_reminder, divisor)
        else: operated_reminder = sum_binary(new_reminder, inverse_binary(divisor))
        return operated_reminder, new_dividend

    def _generate_and_show_plots(self):
        """Генерирует и отображает графики производительности."""
        print("\n--- Generating Performance Graphs ---")
        n_sim = self._count
        r_sim = self._num_tasks

        if n_sim <= 0 or r_sim <= 0:
             print("Cannot generate plots for n=0 or r=0.")
             return

        print(f"Plotting theoretical performance for n={n_sim}, r={r_sim} (without highlighting)")

        # Параметры для осей и кривых
        r_plot_limit = max(20, r_sim + 5)
        n_plot_limit = max(n_sim + 2, 8)
        r_curves_limit = max(8, r_sim + 1) # Сколько кривых r=... показать

        # Вызов функций построения графиков (обновленных)
        plot_ky_vs_r(n_sim, r_sim, r_max_plot=r_plot_limit)
        plot_e_vs_r(n_sim, r_sim, r_max_plot=r_plot_limit)
        plot_ky_vs_n(n_sim, r_sim, n_max_plot=n_plot_limit, r_max_curves=r_curves_limit)
        plot_e_vs_n(n_sim, r_sim, n_max_plot=n_plot_limit, r_max_curves=r_curves_limit)

        plt.show() # Показываем все созданные окна
        print("--- End of Plotting ---")


# --- Точка входа ---
if __name__ == "__main__":
    try:
        dividends_in = input("Enter space-separated dividends: ").split()
        divisors_in = input("Enter space-separated divisors: ").split()
        n_digits_in = int(input("Enter number of binary digits (pipeline stages, n): "))

        conveyor_instance = Conveyer(dividends_in, divisors_in, n_digits_in)

    except ValueError:
        print("Error: Number of binary digits must be an integer.")
        exit(1)
    # MyException обрабатывается внутри _check_values или напрямую
    except MyException as e:
         print(f"Error: {e}") # Сообщение об ошибке уже было выведено в _check_values
         exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        exit(1)