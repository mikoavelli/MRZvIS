# Индивидуальная лабораторная работа 2 по дисциплине МРЗвИС вариант 7
# (~) = /2\\; /~\\ = /1\\; x~>y = max({1-x}\\/{y})
# Выполнена студентом группы X БГУИР X X X
# Использованзоватьные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

import math
import random

import matplotlib.pyplot as plt

ALFABET = [str(i) for i in range(10)]
A, B, E, G, C = [], [], [], [], []
T_parallel = 0
p, q, m = 0, 0, 0
call_sum_count, call_mult_count, call_diff_count, call_compare_count = 0, 0, 0, 0
t_sum, t_mult, t_diff, t_comparison, n, r = 1, 1, 1, 1, 1, 1
Lavg, Tavg = 0, 0


def sum_(a, b):
    global call_sum_count
    call_sum_count += 1
    return a + b


def mult_(a, b):
    global call_mult_count
    call_mult_count += 1
    return a * b


def diff_(a, b):
    global call_diff_count
    call_diff_count += 1
    return a - b


def compare_(a, b, max_or_min):
    global call_compare_count
    call_compare_count += 1
    if max_or_min:
        return max(a, b)
    else:
        return min(a, b)


def is_input_invalid(user_input):
    for i in user_input:
        if i not in ALFABET:
            return 1
    return 0


def print_matrix(matr, name=""):
    print(name)
    for row in matr:
        string = "   "
        for col in row:
            string += str(col) + "  "
        print(string)


def fill_matrix(m, p, q):
    global A, B, E, G
    A = [[round(random.uniform(-1, 1.001), 3) for _ in range(m)] for i in range(p)]
    B = [[round(random.uniform(-1, 1.001), 3) for _ in range(q)] for i in range(m)]
    E = [[round(random.uniform(-1, 1.001), 3) for _ in range(m)] for i in range(1)]
    G = [[round(random.uniform(-1, 1.001), 3) for _ in range(q)] for i in range(p)]


# def fill_matrix(m, p, q):
#     global A, B, E, G
#     A = [[0.594, -0.99],
#          [0.948, 0.96]]
#     B = [[0.297, -0.069],
#          [0.008, -0.801]]
#     E = [[0.777, -0.921]]
#     G = [[0.149, 0.714],
#          [0.868, -0.394]]


def find_Tavg():
    Tavg = 0
    Tavg += p * q * m * (3 * (t_diff + t_comparison) + 7 * t_mult + 3 * t_diff + 2 * t_sum)  # f[i][j][k]
    Tavg += p * q * m * t_comparison  # d[i][j][k]
    Tavg += p * q * (m - 1) * t_mult  # /~\k
    Tavg += p * q * ((m + 1) * t_diff + (m - 1) * t_mult)  # \~/k
    Tavg += p * q * (7 * t_mult + 2 * t_sum + 3 * t_diff + t_mult)  # c[i][j]
    return Tavg


def find_C(x, y, m):
    global C

    def find_compose(a, b):  # a * b
        return mult_(a, b)

    def find_tnorm(a, b):  # min(a,b)
        return compare_(a, b, 0)

    def find_impl(a, b):  # max(1 - a,b)
        return compare_(diff_(1, a), b, 1)

    def find_kf(i, j):
        # f = a_to_b*(2*E[0][k]-1)*E[0][k] + b_to_a*(1+(4*a_to_b-2)*E[0][k])*(1-E[0][k])
        global \
            A, \
            B, \
            E, \
            G, \
            call_mult_count, \
            call_diff_count, \
            call_sum_count, \
            call_compare_count, \
            n, \
            T_parallel, \
            p, \
            q, \
            m, \
            Tavg
        mult_arr = []
        T_parallel_old = T_parallel

        for k in range(m):
            a_to_b = find_impl(A[i][k], B[k][j])
            b_to_a = find_impl(B[k][j], A[i][k])
            temp1 = mult_(mult_(a_to_b, diff_(mult_(2, E[0][k]), 1)), E[0][k])
            temp2 = mult_(
                mult_(b_to_a, sum_(1, (mult_(diff_(mult_(4, a_to_b), 2), E[0][k])))),
                diff_(1, E[0][k]),
            )
            mult_arr.append(sum_(temp1, temp2))

            T_parallel += math.ceil(3 / n) * t_diff
            T_parallel += 1 * t_mult
            T_parallel += math.ceil(2 / n) * t_comparison
            T_parallel += 1 * t_mult
            T_parallel += math.ceil(2 / n) * t_diff
            T_parallel += math.ceil(2 / n) * t_mult
            T_parallel += 1 * t_mult
            T_parallel += 1 * t_sum
            T_parallel += 1 * t_mult
            T_parallel += 1 * t_mult
            T_parallel += 1 * t_sum

        if 6 <= n <= m * 3:
            n_actual = n - n % 3  # будет задействоваться максимальное n кратное 3
            count = math.ceil((m * 3) / n_actual)  # сколько должно выполниться последовательных операций
            temp = (T_parallel - T_parallel_old) / m  # сколько по времени одна итерация
            T_parallel = T_parallel - (m - count) * temp  # отнимаем операции, которые можно распараллелить
        elif n >= m * 3:
            temp = (T_parallel - T_parallel_old) / m
            T_parallel = T_parallel_old + temp

        kf = mult_arr[0]
        for i_mult in range(1, len(mult_arr)):
            kf = mult_(kf, mult_arr[i_mult])

        T_parallel += math.ceil(m - 1) * t_mult

        return kf

    def find_kd(i, j):
        nonlocal m
        global A, B, call_mult_count, call_diff_count, call_sum_count, call_compare_count, n, T_parallel, Tavg, q, p
        multipl_arr = []
        old_Tn = T_parallel

        for k in range(m):
            temp1 = find_tnorm(A[i][k], B[k][j])
            temp2 = diff_(1, temp1)
            multipl_arr.append(temp2)
            T_parallel += 1 * t_comparison
            T_parallel += 1 * t_diff

        if 2 <= n <= m * 1:
            new_n = n - n % 1  # будет задействоваться максимальное n кратное 1
            count = math.ceil((m * 1) / new_n)  # сколько должно выполниться последоватеьных операций
            temp = (T_parallel - old_Tn) / m  # сколько по времени одна итерация
            T_parallel = T_parallel - (m - count) * temp  # отнимаем операции, которые можно распараллелить
        elif n >= m * 1:
            temp = (T_parallel - old_Tn) / m
            T_parallel = old_Tn + temp

        dd_res = multipl_arr[0]
        for i_mult in range(1, len(multipl_arr)):
            dd_res = mult_(dd_res, multipl_arr[i_mult])
        dd = diff_(1, dd_res)

        T_parallel += math.ceil(m - 1) * t_mult
        T_parallel += 1 * t_diff

        return dd

    def find_cij(i, j):
        # cij = f*(3*G[i][j] - 2)*G[i][j] + (d+(4*f_and_d-3*d)*G[i][j])*(1-G[i][j])
        global A, B, E, G, call_sum_count, call_diff_count, call_mult_count, call_compare_count, n, T_parallel
        d = find_kd(i, j)  # with \~/k
        f = find_kf(i, j)  # with /~\k

        f_and_d = find_compose(f, d)
        cij = sum_(
            mult_(mult_(f, diff_(mult_(3, G[i][j]), 2)), G[i][j]),
            mult_(
                sum_(d, mult_(diff_(mult_(4, f_and_d), mult_(3, d)), G[i][j])),
                diff_(1, G[i][j]),
            ),
        )
        T_parallel += 1 * t_mult
        T_parallel += math.ceil(3 / n) * t_mult
        T_parallel += math.ceil(3 / n) * t_diff
        T_parallel += math.ceil(2 / n) * t_mult
        T_parallel += 1 * t_mult
        T_parallel += math.ceil(2 / n) * t_mult
        T_parallel += 1 * t_sum

        return cij

    C = [[find_cij(i, j) for j in range(y)] for i in range(x)]


def main():
    global p, q, m, T_parallel, n
    while 1:
        m = input("m = ")
        p = input("p = ")
        q = input("q = ")
        n = input("n = ")
        # t_sum = int(input("Время операции суммирования: "))
        # t_mult = int(input("Время операции умножения: "))
        # t_diff = int(input("Время операции вычитания: "))
        # t_comparison = int(input("Время операции сравнения: "))
        print("\n")
        if is_input_invalid(m + p + q + n):
            print("Некорректный ввод")
            continue
        elif int(n) == 0 or int(p) == 0 or int(m) == 0 or int(q) == 0:
            print("Введите значения больше 0")
            continue
        else:
            p = int(p)
            q = int(q)
            m = int(m)
            fill_matrix(m, p, q)
            n = int(n)
            find_C(int(p), int(q), int(m))
            break

    T_sequential = (
        call_mult_count * t_mult + call_diff_count * t_diff + call_sum_count * t_sum + call_compare_count * t_comparison
    )
    Ky = T_sequential / T_parallel
    e = Ky / n
    r = p * q + p * m + q * m + 1 * m + p * q
    Tavg = find_Tavg()
    Lavg = Tavg / r
    D = T_parallel / Lavg

    print_matrix(A, "\nA:")
    print_matrix(B, "\nB:")
    print_matrix(E, "\nE:")
    print_matrix(G, "\nG:")
    print_matrix(C, "\nC:")

    print("\nParametrs:")
    print("T_sequential = " + str(T_sequential))
    print("Tn = " + str(T_parallel))
    print("r = " + str(r))
    print("Ky = " + str(Ky))
    print("e = " + str(e))
    print("Lsum = " + str(T_parallel))
    print("Lavg = " + str(Lavg))
    print("D = " + str(D))


def main_graphicsKr():
    global p, q, m, T_parallel, call_sum_count, call_mult_count, call_diff_count, call_compare_count, n, Tavg
    t_mult, t_diff, t_sum, t_comparison = 1, 1, 1, 1  # задаём времена операций
    ky_n10 = []
    ky_n7 = []
    r_vals = []

    for i in range(20):
        # --- n = 10 ---
        T_parallel, Tavg = 0, 0
        call_sum_count, call_mult_count, call_diff_count, call_compare_count = (
            0,
            0,
            0,
            0,
        )
        m = p = q = i + 1
        n = 10
        fill_matrix(m, p, q)
        find_C(p, q, m)
        r = p * q + q * m + p * m + m + p * q
        T_sequential = (
            call_mult_count * t_mult
            + call_diff_count * t_diff
            + call_sum_count * t_sum
            + call_compare_count * t_comparison
        )
        Ky = T_sequential / T_parallel
        ky_n10.append(Ky)
        print(f"n = {n}, m = {m}, r = {r}, T_sequential = {T_sequential}, Tn = {T_parallel}, Ky = {Ky}")
        r_vals.append(r)

    for i in range(20):
        # --- n = 7 ---
        T_parallel, Tavg = 0, 0
        call_sum_count, call_mult_count, call_diff_count, call_compare_count = (
            0,
            0,
            0,
            0,
        )
        m = p = q = i + 1
        n = 7
        fill_matrix(m, p, q)
        find_C(p, q, m)
        # r = p * q + q * m + p * m + m + p * q
        T_sequential = (
            call_mult_count * t_mult
            + call_diff_count * t_diff
            + call_sum_count * t_sum
            + call_compare_count * t_comparison
        )
        Ky = T_sequential / T_parallel
        print(f"n = {n}, m = {m}, r = {r}, T_sequential = {T_sequential}, Tn = {T_parallel}, Ky = {Ky}")
        ky_n7.append(Ky)

    plt.figure(figsize=(10, 5))
    plt.plot(r_vals, ky_n10, "k", label="n = 10", linewidth=2)
    plt.plot(r_vals, ky_n7, label="n = 7", linewidth=3)
    plt.xlabel("r", fontsize=14)
    plt.ylabel("Ky(r)", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best", fontsize=12)
    plt.show()


def main_graphicsKyn():
    global p, q, m, T_parallel, call_sum_count, call_mult_count, call_diff_count, call_compare_count, n, Tavg
    t_mult, t_diff, t_sum, t_comparison = 1, 1, 1, 1
    x = []
    ky_40 = []

    for n in range(1, 51):
        found = False
        for p in range(1, 11):
            for q in range(1, 11):
                for m in range(1, 11):
                    r = p * q + q * m + p * m + m + p * q
                    if r == 40:
                        T_parallel, Tavg = 0, 0
                        (
                            call_sum_count,
                            call_mult_count,
                            call_diff_count,
                            call_compare_count,
                        ) = (0, 0, 0, 0)
                        fill_matrix(m, p, q)
                        find_C(p, q, m)
                        T_sequential = (
                            call_mult_count * t_mult
                            + call_diff_count * t_diff
                            + call_sum_count * t_sum
                            + call_compare_count * t_comparison
                        )
                        Ky = T_sequential / T_parallel
                        ky_40.append(Ky)
                        x.append(n)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    x2 = []  # значения n для r = 33
    ky_33 = []

    for n in range(1, 51):
        found = False
        for p in range(1, 11):
            for q in range(1, 11):
                for m in range(1, 11):
                    r = p * q + q * m + p * m + m + p * q
                    if r == 33:
                        T_parallel, Tavg = 0, 0
                        (
                            call_sum_count,
                            call_mult_count,
                            call_diff_count,
                            call_compare_count,
                        ) = (0, 0, 0, 0)
                        fill_matrix(m, p, q)
                        find_C(p, q, m)
                        T_sequential = (
                            call_mult_count * t_mult
                            + call_diff_count * t_diff
                            + call_sum_count * t_sum
                            + call_compare_count * t_comparison
                        )
                        Ky = T_sequential / T_parallel
                        ky_33.append(Ky)
                        x2.append(n)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # Построение графика
    plt.figure(figsize=(10, 5))
    plt.plot(x, ky_40, "k", label="r = 40", linewidth=2)
    plt.plot(x2, ky_33, label="r = 33", linewidth=3)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("Ky(n)", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best", fontsize=12)
    plt.show()


def main_graphicsEr():
    global p, q, m, T_parallel, call_sum_count, call_mult_count, call_diff_count, call_compare_count, n, Tavg
    t_mult, t_diff, t_sum, t_comparison = 1, 1, 1, 1  # задаём времена операций
    e_n7 = []
    e_n10 = []
    r_vals = []

    for i in range(20):
        # --- n = 10 ---
        T_parallel, Tavg = 0, 0
        call_sum_count, call_mult_count, call_diff_count, call_compare_count = (
            0,
            0,
            0,
            0,
        )
        m = p = q = i + 1
        n = 10
        fill_matrix(m, p, q)
        find_C(p, q, m)
        r = p * q + q * m + p * m + m + p * q
        T_sequential = (
            call_mult_count * t_mult
            + call_diff_count * t_diff
            + call_sum_count * t_sum
            + call_compare_count * t_comparison
        )
        Ky = T_sequential / T_parallel
        e = Ky / n
        e_n10.append(e)
        r_vals.append(r)

    for i in range(20):
        # --- n = 7 ---
        T_parallel, Tavg = 0, 0
        call_sum_count, call_mult_count, call_diff_count, call_compare_count = (
            0,
            0,
            0,
            0,
        )
        m = p = q = i + 1
        n = 7
        fill_matrix(m, p, q)
        find_C(p, q, m)
        r = p * q + q * m + p * m + m + p * q
        T_sequential = (
            call_mult_count * t_mult
            + call_diff_count * t_diff
            + call_sum_count * t_sum
            + call_compare_count * t_comparison
        )
        Ky = T_sequential / T_parallel
        e = Ky / n
        e_n7.append(e)

    plt.figure(figsize=(10, 5))
    plt.plot(r_vals, e_n10, "k", label="n = 10", linewidth=2)
    plt.plot(r_vals, e_n7, label="n = 7", linewidth=3)
    plt.xlabel("r", fontsize=14)
    plt.ylabel("e(r)", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best", fontsize=12)
    plt.show()


def main_graphicsEn():
    global p, q, m, T_parallel, call_sum_count, call_mult_count, call_diff_count, call_compare_count, n, Tavg
    t_mult, t_diff, t_sum, t_comparison = 1, 1, 1, 1
    x = []  # значения n для r = 40
    e_40 = []

    for n in range(1, 51):
        found = False
        for p in range(1, 11):
            for q in range(1, 11):
                for m in range(1, 11):
                    r = p * q + q * m + p * m + m + p * q
                    if r == 40:
                        T_parallel, Tavg = 0, 0
                        (
                            call_sum_count,
                            call_mult_count,
                            call_diff_count,
                            call_compare_count,
                        ) = (0, 0, 0, 0)
                        fill_matrix(m, p, q)
                        find_C(p, q, m)
                        T_sequential = (
                            call_mult_count * t_mult
                            + call_diff_count * t_diff
                            + call_sum_count * t_sum
                            + call_compare_count * t_comparison
                        )
                        Ky = T_sequential / T_parallel if T_parallel != 0 else 0
                        e = Ky / n
                        e_40.append(e)
                        x.append(n)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    x2 = []  # значения n для r = 33
    e_33 = []

    for n in range(1, 51):
        found = False
        for p in range(1, 11):
            for q in range(1, 11):
                for m in range(1, 11):
                    r = p * q + q * m + p * m + m + p * q
                    if r == 33:
                        T_parallel, Tavg = 0, 0
                        (
                            call_sum_count,
                            call_mult_count,
                            call_diff_count,
                            call_compare_count,
                        ) = (0, 0, 0, 0)
                        fill_matrix(m, p, q)
                        find_C(p, q, m)
                        T_sequential = (
                            call_mult_count * t_mult
                            + call_diff_count * t_diff
                            + call_sum_count * t_sum
                            + call_compare_count * t_comparison
                        )
                        Ky = T_sequential / T_parallel if T_parallel != 0 else 0
                        e = Ky / n
                        e_33.append(e)
                        x2.append(n)
                        found = True
                        break
                if found:
                    break
            if found:
                break

    # Построение графика
    plt.figure(figsize=(10, 5))
    plt.plot(x, e_40, "k", label="r = 40", linewidth=2)
    plt.plot(x2, e_33, label="r = 33", linewidth=3)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("e(n)", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best", fontsize=12)
    plt.show()


def main_graphicsDr():
    global p, q, m, T_parallel, call_sum_count, call_mult_count, call_diff_count, call_compare_count, n, Tavg
    x = []
    d_ = []
    ky_ = []

    for i in range(10):
        T_parallel, Tavg = 0, 0
        call_sum_count, call_mult_count, call_diff_count, call_compare_count = (
            0,
            0,
            0,
            0,
        )
        m = i + 1
        p = i + 1
        q = i + 1
        n = 10
        fill_matrix(int(m), int(p), int(q))
        find_C(int(p), int(q), int(m))
        r = p * q + q * m + p * m + 1 * m + p * q
        T_sequential = (
            call_mult_count * t_mult
            + call_diff_count * t_diff
            + call_sum_count * t_sum
            + call_compare_count * t_comparison
        )
        Ky = T_sequential / T_parallel
        Ky / n
        Tavg = find_Tavg()
        Lavg = Tavg / r
        D = T_parallel / Lavg
        ky_.append(Ky)
        d_.append(D)
        x.append(r)
    y2 = []
    for i in range(10):
        T_parallel, Tavg = 0, 0
        call_sum_count, call_mult_count, call_diff_count, call_compare_count = (
            0,
            0,
            0,
            0,
        )
        m = i + 1
        p = i + 1
        q = i + 1
        n = 7
        fill_matrix(int(m), int(p), int(q))
        find_C(int(p), int(q), int(m))
        r = p * q + q * m + p * m + 1 * m + p * q
        T_sequential = (
            call_mult_count * t_mult
            + call_diff_count * t_diff
            + call_sum_count * t_sum
            + call_compare_count * t_comparison
        )
        Ky = T_sequential / T_parallel
        Ky / n
        Tavg = find_Tavg()
        Lavg = Tavg / r
        D = T_parallel / Lavg
        ky_.append(Ky)
        y2.append(D)

    y = d_
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "k", label="n = 10", linewidth=2)
    plt.plot(x, y2, label="n = 7", linewidth=3)
    plt.xlabel("r", fontsize=14)
    plt.ylabel("D(r)", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best", fontsize=12)
    plt.show()


def main_graphicsDn():
    global p, q, m, T_parallel, call_sum_count, call_mult_count, call_diff_count, call_compare_count, n, Tavg
    x = []
    d_ = []
    ky_ = []

    for n in range(1, 51):
        for p in range(1, 11):
            for q in range(1, 11):
                for m in range(1, 11):
                    T_parallel, Tavg = 0, 0
                    (
                        call_sum_count,
                        call_mult_count,
                        call_diff_count,
                        call_compare_count,
                    ) = (0, 0, 0, 0)
                    r = p * q + q * m + p * m + 1 * m + p * q
                    if r == 33:
                        fill_matrix(int(m), int(p), int(q))
                        find_C(int(p), int(q), int(m))
                        T_sequential = (
                            call_mult_count * t_mult
                            + call_diff_count * t_diff
                            + call_sum_count * t_sum
                            + call_compare_count * t_comparison
                        )
                        Ky = T_sequential / T_parallel
                        Ky / n
                        Tavg = find_Tavg()
                        Lavg = Tavg / r
                        D = T_parallel / Lavg
                        ky_.append(Ky)
                        d_.append(D)
                        x.append(n)
                        break  # Нашли подходящие p, q, m, выходим из цикла
                else:
                    continue
                break
            else:
                continue
            break

    y2 = []
    x2 = []
    for n in range(1, 51):
        for p in range(1, 11):
            for q in range(1, 11):
                for m in range(1, 11):
                    T_parallel, Tavg = 0, 0
                    (
                        call_sum_count,
                        call_mult_count,
                        call_diff_count,
                        call_compare_count,
                    ) = (0, 0, 0, 0)
                    r = p * q + q * m + p * m + 1 * m + p * q
                    if r == 40:
                        fill_matrix(int(m), int(p), int(q))
                        find_C(int(p), int(q), int(m))
                        T_sequential = (
                            call_mult_count * t_mult
                            + call_diff_count * t_diff
                            + call_sum_count * t_sum
                            + call_compare_count * t_comparison
                        )
                        Ky = T_sequential / T_parallel
                        Ky / n
                        Tavg = find_Tavg()
                        Lavg = Tavg / r
                        D = T_parallel / Lavg
                        ky_.append(Ky)
                        y2.append(D)
                        x2.append(n)
                        break
                else:
                    continue
                break
            else:
                continue
            break

    y = d_
    plt.figure(figsize=(10, 5))
    plt.plot(x, y, "k", label="r = 33", linewidth=2)
    plt.plot(x2, y2, label="r = 40", linewidth=3)
    plt.xlabel("n", fontsize=14)
    plt.ylabel("D(n)", fontsize=14)
    plt.grid(True)
    plt.legend(loc="best", fontsize=12)
    plt.show()


main()

# main_graphicsKr()
# main_graphicsKyn()
# main_graphicsEr()
# main_graphicsEn()
# main_graphicsDr()
# main_graphicsDn()
