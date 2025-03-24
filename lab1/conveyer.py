# Non-restoring division

import os
from binary_logic import to_binary, from_binary, sum_binary, inverse_binary


class MyException(Exception):
    def __init__(self, m):
        self.message = m

    def __str__(self):
        return self.message


class Conveyer:
    def __init__(self,
                 list_dividend: list[str, ...],
                 list_divisor: list[str, ...],
                 num_of_bin_digits: int
                 ) -> None:
        self._dividends, self._divisors = self._check_values(list_dividend, list_divisor, num_of_bin_digits)
        self._count: int = num_of_bin_digits
        self._generate_conveyer()
        self._start_conveyer()

    def _check_values(self,
                      list_dividend: list[str, ...],
                      list_divisor: list[str, ...],
                      num_of_bin_digits: int
                      ) -> tuple[list[int, ...], list[int, ...]]:
        try:
            dividends: list[int, ...] = [int(num) for num in list_dividend]
            divisors: list[int, ...] = [int(num) for num in list_divisor]

            if num_of_bin_digits <= 0:
                raise MyException("Count must be greater than 0")

            if len(dividends) != len(divisors):
                raise MyException("The number of dividends does not equal the number of divisors")

            for i in range(len(dividends)):
                if dividends[i] < 0:
                    raise MyException(f"The dividend {dividends[i]} is negative")
                if dividends[i] > 2 ** num_of_bin_digits - 1:
                    raise MyException("The dividend is too large")
                if divisors[i] <= 0:
                    raise MyException(f"The divisor {divisors[i]} is not positive")
                if divisors[i] > 2 ** num_of_bin_digits - 1:
                    raise MyException("The divisor is too large")
        except MyException as exception:
            print(f"{exception}. Abort program")
            exit(0)

        return dividends, divisors

    def _generate_conveyer(self) -> None:
        self._conveyer: dict = {'queue': tuple(zip(self._dividends, self._divisors))}

        for i in range(self._count):
            self._conveyer[f"step{i + 1}"]: dict[str: str, ...] = dict()

        self._conveyer['result']: tuple[dict[str: str, str: str, str: str]] = None
        self._conveyer['queue_backup']: tuple[list[int], ...] = tuple(zip(self._dividends, self._divisors))
        self._conveyer['tact']: int = 0

    def _start_conveyer(self) -> None:
        while self._conveyer['tact'] != self._count + len(self._dividends):
            self._print_conveyer()
            self._next()
            self._conveyer['tact'] += 1
            input()
        else:
            self._conveyer['tact'] = 0
            self._print_conveyer()

    def _print_conveyer(self) -> None:
        print(f"Tact: {self._conveyer['tact']}")
        print("Enter queue:")
        os.system('clear')

        if self._conveyer['tact'] or self._conveyer['queue']:
            print(f"Tact: {self._conveyer['tact']}")
            for dividend, divisor in self._conveyer['queue']:
                print(f"{dividend} / {divisor}")
        else:
            print(f"Total tact: {len(self._conveyer['queue_backup']) + self._count - 1}")
            for dividend, divisor in self._conveyer['queue_backup']:
                print(f"{dividend} / {divisor}")

        for i in range(1, self._count + 1):
            print(f"Step {i}")

            step = self._conveyer[f"step{i}"]
            if not step:
                print()
            else:
                print(f" Reminder: {step['reminder']}")
                print(f" Dividend: {step['dividend']}")
                print(f" Divisor: {step['divisor']}")

        if self._conveyer['result'] is not None:
            print("Result")

            for result in self._conveyer['result']:
                print(f" Reminder: {result['reminder']} - {from_binary(result['reminder'])}", end=' ')
                print(f" Quotient: {result['dividend']} - {from_binary(result['dividend'])}")

        print('-' * 20)

    def _next(self) -> None:
        if self._conveyer[f"step{self._count}"]:
            self._calculate_last_step()

        for i in range(self._count - 1, 0, -1):
            step: dict[str: str, str: str, str: str] = self._conveyer[f"step{i}"]
            if step:
                self._conveyer[f"step{i + 1}"]: dict[str: str, ...] = self._calculate_step(step)
                self._conveyer[f"step{i}"] = None

        if self._conveyer['queue']:
            dividend, divisor = self._conveyer['queue'][0]
            self._conveyer['queue'] = self._conveyer['queue'][1:]
            step: dict[str: str, str: str, str: str] = {
                'reminder': '0' * (self._count + 1),
                'dividend': to_binary(dividend, self._count),
                'divisor': to_binary(divisor, self._count + 1)
            }
            self._conveyer['step1'] = self._calculate_step(step)

    def _calculate_last_step(self) -> None:
        if self._conveyer[f"step{self._count}"]['reminder'][0] == '1':
            self._conveyer[f"step{self._count}"] = {
                'reminder': sum_binary(
                    self._conveyer[f'step{self._count}']['reminder'],
                    self._conveyer[f'step{self._count}']['divisor']),
                'dividend': self._conveyer[f'step{self._count}']['dividend'],
                'divisor': self._conveyer[f'step{self._count}']['divisor']
            }
        if self._conveyer[f'result'] is None:
            self._conveyer[f'result'] = (self._conveyer[f'step{self._count}'],)
        else:
            self._conveyer[f'result'] = self._conveyer[f'result'] + (self._conveyer[f'step{self._count}'],)
        self._conveyer[f'step{self._count}'] = None

    def _calculate_step(self, step: dict[str: str, str: str, str: str]):
        step['reminder'], step['dividend'] = self._shift_and_operation(
            step['reminder'],
            step['dividend'],
            step['divisor'],
            is_operation_sum=(step['reminder'][0] == '1')
        )
        step['dividend'] = step['dividend'][:-1] + '0' \
            if step['reminder'][0] == '1' \
            else step['dividend'][:-1] + '1'
        return step

    def _shift_and_operation(self, reminder: str, dividend: str, divisor: str, is_operation_sum: bool) -> str:
        reminder, dividend = reminder[1:] + dividend[0], dividend[1:] + '_'
        reminder = sum_binary(reminder, divisor) \
            if is_operation_sum \
            else sum_binary(reminder, inverse_binary(divisor))
        return reminder, dividend


Conveyer(
    input("Enter space-separated dividends: ").split(),
    input("Enter space-separated divisors: ").split(),
    int(input("Enter number of binary digits: "))
)
