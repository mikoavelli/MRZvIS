# Индивидуальная лабораторная работа 1 по дисциплине МРЗвИС вариант 12
# Частное пары 6-разрядных чисел без восстановления частичного остатка
# Выполнена студентом группы X БГУИР X X X
# Использованзоватьные источники:
# Формальные модели обработки информации и параллельные модели решения задач.
# Практикум: учебно-методическое пособие / В.П.Ивашенко. – Минск: БГУИР, 2020.

def to_binary(int_num: int, num_of_bin_digits: int) -> str:
    binary_num: str = ""

    if int_num == 0:
        return binary_num.zfill(num_of_bin_digits)

    while int_num != 0:
        binary_num = str(int_num % 2) + binary_num
        int_num //= 2

    return binary_num.zfill(num_of_bin_digits)


def from_binary(binary_num: str) -> int:
    result: int = 0
    binary_num = binary_num[::-1]
    for i in range(len(binary_num)):
        result += int(binary_num[i]) * 2**i
    return result


def sum_binary(binary_num1: str, binary_num2: str) -> str:
    binary_num1: str = binary_num1[::-1]
    binary_num2: str = binary_num2[::-1]
    result: str = ""
    add_term: int = 0

    for i in range(len(binary_num1)):
        curr_sum: int = add_term + int(binary_num1[i]) + int(binary_num2[i])
        if curr_sum >= 2:
            add_term = 1
            result += str(curr_sum - 2)
        else:
            add_term = 0
            result += str(curr_sum)

    return result[::-1]


def inverse_binary(binary_num: str) -> str:
    result: str = ""
    for number in binary_num:
        result += "1" if number == "0" else "0"
    return sum_binary(result, "".zfill(len(binary_num) - 1) + "1")
