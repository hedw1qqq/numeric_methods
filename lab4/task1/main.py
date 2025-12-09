import math
from typing import List, Tuple, Callable

# --- Типы данных ---
Vector = List[float]
DerivativeFunc = Callable[[float, Vector], Vector]
ErrorTuple = Tuple[float, float, float]


# --- Векторная арифметика ---
def vec_add(v1: Vector, v2: Vector) -> Vector:
    return [a + b for a, b in zip(v1, v2)]


def vec_scale(v: Vector, s: float) -> Vector:
    return [a * s for a in v]


# --- Генерация сетки ---
def generate_grid(x_start: float, x_end: float, h: float) -> List[float]:
    """Генерирует равномерную сетку узлов."""
    steps = int(round((x_end - x_start) / h))
    return [x_start + i * h for i in range(steps + 1)]


# --- Уравнения задачи ---
def exact_solution(x: float) -> float:
    """Точное решение: y = x*sin(x) + cos(x)"""
    return x * math.sin(x) + math.cos(x)


def ode_system(x: float, Y: Vector) -> Vector:
    """
    Система: y' = z, z' = 2cos(x) - y
    """
    y, z = Y
    return [z, 2 * math.cos(x) - y]


# --- Численные методы ---

def euler_method(
        func: DerivativeFunc, xs: List[float], y0: Vector
) -> List[Vector]:
    """Метод Эйлера на заданной сетке."""
    ys = [y0]
    curr_y = y0

    for i in range(len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x  # Локальный шаг

        diff = func(curr_x, curr_y)
        curr_y = vec_add(curr_y, vec_scale(diff, h))
        ys.append(curr_y)
    return ys


def runge_kutta_4(
        func: DerivativeFunc, xs: List[float], y0: Vector
) -> List[Vector]:
    """Метод Рунге-Кутты 4-го порядка на заданной сетке."""
    ys = [y0]
    curr_y = y0

    for i in range(len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x

        k1 = vec_scale(func(curr_x, curr_y), h)

        y_k2 = vec_add(curr_y, vec_scale(k1, 0.5))
        k2 = vec_scale(func(curr_x + 0.5 * h, y_k2), h)

        y_k3 = vec_add(curr_y, vec_scale(k2, 0.5))
        k3 = vec_scale(func(curr_x + 0.5 * h, y_k3), h)

        y_k4 = vec_add(curr_y, k3)
        k4 = vec_scale(func(curr_x + h, y_k4), h)

        sum_k = vec_add(
            vec_add(k1, vec_scale(k2, 2.0)),
            vec_add(vec_scale(k3, 2.0), k4)
        )
        delta = vec_scale(sum_k, 1.0 / 6.0)
        curr_y = vec_add(curr_y, delta)
        ys.append(curr_y)
    return ys


def adams_bashforth_4(
        func: DerivativeFunc, xs: List[float], y0: Vector
) -> List[Vector]:
    """Метод Адамса 4-го порядка на заданной сетке."""
    # Разгон: первые 4 точки берем из РК4
    # Мы просто запускаем РК4 на всей сетке, но берем только первые 4 значения
    # Важно: это работает корректно только если xs имеет хотя бы 4 точки
    rk_full = runge_kutta_4(func, xs[:4], y0)

    ys = []
    ys.extend(rk_full)  # [y0, y1, y2, y3]

    # Основной цикл (начиная с 4-го шага, чтобы найти y4 и т.д.)
    for i in range(3, len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x

        # Значения производных в предыдущих узлах
        f0 = func(xs[i], ys[i])  # f_i
        f1 = func(xs[i - 1], ys[i - 1])  # f_{i-1}
        f2 = func(xs[i - 2], ys[i - 2])  # f_{i-2}
        f3 = func(xs[i - 3], ys[i - 3])  # f_{i-3}

        # Формула Адамса
        combo = vec_add(
            vec_add(vec_scale(f0, 55), vec_scale(f1, -59)),
            vec_add(vec_scale(f2, 37), vec_scale(f3, -9))
        )
        delta = vec_scale(combo, h / 24.0)

        new_y = vec_add(ys[i], delta)
        ys.append(new_y)
    return ys


# --- Логика расчета ошибок ---

def calculate_point_errors(
        y_h: float,
        y_h2: float,
        y_exact: float,
        p: int
) -> ErrorTuple:
    """
    Считает ошибки в конкретной точке.
    Возвращает кортеж: (Рунге-Ромберг, Абсолютная, Относительная %)
    """
    rr_error = abs(y_h - y_h2) / (2 ** p - 1)
    abs_error = abs(y_h - y_exact)
    if abs(y_exact) > 1e-14:
        rel_error_pct = (abs_error / abs(y_exact)) * 100.0
    else:
        rel_error_pct = 0.0
    return rr_error, abs_error, rel_error_pct


def main():
    # Параметры
    x_start, x_finish, h = 0.0, 1.0, 0.1
    y_start = [1.0, 0.0]

    # 1. Генерация сеток
    xs = generate_grid(x_start, x_finish, h)
    xs_h2 = generate_grid(x_start, x_finish, h / 2.0)

    # 2. Решения на основной сетке
    res_eu = euler_method(ode_system, xs, y_start)
    res_rk = runge_kutta_4(ode_system, xs, y_start)
    res_ad = adams_bashforth_4(ode_system, xs, y_start)

    # 3. Решения на мелкой сетке (для Рунге-Ромберга)
    res_eu_h2 = euler_method(ode_system, xs_h2, y_start)
    res_rk_h2 = runge_kutta_4(ode_system, xs_h2, y_start)
    res_ad_h2 = adams_bashforth_4(ode_system, xs_h2, y_start)

    # --- Вывод ---
    h1 = (
        f"{'x':^4} | {'РЕШЕНИЯ (y)':^39} || "
        f"{'ОШИБКА РУНГЕ-РОМБЕРГА':^32} | "
        f"{'АБСОЛЮТНАЯ ПОГРЕШНОСТЬ':^32} | "
        f"{'ОТНОСИТЕЛЬНАЯ (%)':^30}"
    )
    h2 = (
        f"{'':^4} | {'Точное':^9} {'Эйлер':^9} {'РК':^9} {'Адамс':^9} || "
        f"{'Эйлер':^10} {'РК':^10} {'Адамс':^10} | "
        f"{'Эйлер':^10} {'РК':^10} {'Адамс':^10} | "
        f"{'Эйлер':^9} {'РК':^10} {'Адамс':^9}"
    )

    print("=" * len(h1))
    print(h1)
    print("-" * len(h1))
    print(h2)
    print("=" * len(h1))

    for i in range(len(xs)):
        x_curr = xs[i]
        y_ex = exact_solution(x_curr)

        # Значения y (первая компонента вектора состояния)
        y_eu, y_rk, y_ad = res_eu[i][0], res_rk[i][0], res_ad[i][0]

        # Значения на мелкой сетке (индекс i*2)
        y_eu_small = res_eu_h2[i * 2][0]
        y_rk_small = res_rk_h2[i * 2][0]
        y_ad_small = res_ad_h2[i * 2][0]

        # --- Использование функции расчета ошибок ---
        # Эйлер (p=1)
        err_eu = calculate_point_errors(y_eu, y_eu_small, y_ex, p=1)
        # Рунге-Кутта (p=4)
        err_rk = calculate_point_errors(y_rk, y_rk_small, y_ex, p=4)
        # Адамс (p=4)
        err_ad = calculate_point_errors(y_ad, y_ad_small, y_ex, p=4)

        # Распаковка кортежей для печати: (rr, abs, rel)
        rr_eu, abs_eu, rel_eu = err_eu
        rr_rk, abs_rk, rel_rk = err_rk
        rr_ad, abs_ad, rel_ad = err_ad

        print(
            f"{x_curr:^4.1f} | "
            f"{y_ex:^9.5f} {y_eu:^9.5f} {y_rk:^9.5f} {y_ad:^9.5f} || "
            f"{rr_eu:^10.2e} {rr_rk:^10.2e} {rr_ad:^10.2e} | "
            f"{abs_eu:^10.2e} {abs_rk:^10.2e} {abs_ad:^10.2e} | "
            f"{rel_eu:^9.4f} {rel_rk:^10.6f} {rel_ad:^9.6f}"
        )


if __name__ == "__main__":
    main()
