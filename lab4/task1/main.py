import math
from typing import List, Tuple, Callable

import matplotlib.pyplot as plt

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
    ys = [y0]
    curr_y = y0

    for i in range(len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x  # Локальный шаг

        diff = func(curr_x, curr_y)
        curr_y = vec_add(curr_y, vec_scale(diff, h))
        ys.append(curr_y)
    return ys


def improved_euler_first(
        func: DerivativeFunc, xs: List[float], y0: Vector
) -> List[Vector]:
    ys = [y0]
    curr_y = y0

    for i in range(len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x

        # наклон в начале
        k1 = func(curr_x, curr_y)

        y_mid = vec_add(curr_y, vec_scale(k1, h * 0.5))

        # наклон в средин
        k_mid = func(curr_x + 0.5 * h, y_mid)

        curr_y = vec_add(curr_y, vec_scale(k_mid, h))
        ys.append(curr_y)
    return ys


def euler_cauchy_method(
        func: DerivativeFunc, xs: List[float], y0: Vector
) -> List[Vector]:
    ys = [y0]
    curr_y = y0

    for i in range(len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x

        k1 = func(curr_x, curr_y)

        # Предиктор: шаг Эйлера
        y_pred = vec_add(curr_y, vec_scale(k1, h))

        # k2 — производная в конце шага
        k2 = func(curr_x + h, y_pred)

        # Корректор: среднее из k1 и k2
        avg = vec_scale(vec_add(k1, k2), 0.5)
        curr_y = vec_add(curr_y, vec_scale(avg, h))

        ys.append(curr_y)
    return ys


def runge_kutta_4(
        func: DerivativeFunc, xs: List[float], y0: Vector
) -> List[Vector]:
    """Метод Рунге–Кутты 4-го порядка на заданной сетке."""
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
    """Метод Адамса–Башфорта 4-го порядка на заданной сетке."""
    # Разгон: первые 4 точки берём из РК4
    rk_full = runge_kutta_4(func, xs[:4], y0)

    ys = []
    ys.extend(rk_full)  # [y0, y1, y2, y3]

    for i in range(3, len(xs) - 1):
        curr_x = xs[i]
        h = xs[i + 1] - curr_x

        # Значения производных в предыдущих узлах
        f0 = func(xs[i], ys[i])  # f_i
        f1 = func(xs[i - 1], ys[i - 1])  # f_{i-1}
        f2 = func(xs[i - 2], ys[i - 2])  # f_{i-2}
        f3 = func(xs[i - 3], ys[i - 3])  # f_{i-3}

        # Формула Адамса–Башфорта 4-го порядка
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
    rr_error = abs(y_h - y_h2) / (2 ** p - 1)
    abs_error = abs(y_h - y_exact)
    if abs(y_exact) > 1e-14:
        rel_error_pct = (abs_error / abs(y_exact)) * 100.0
    else:
        rel_error_pct = 0.0
    return rr_error, abs_error, rel_error_pct


def plot_method(xs: List[float],
                ys_num: List[float],
                title: str,
                method_label: str,
                color: str = 'r',
                marker: str = 'o') -> None:
    ys_exact = [exact_solution(x) for x in xs]

    plt.figure(figsize=(7, 5))
    plt.plot(xs, ys_exact, 'k-', label='Точное решение')
    plt.plot(xs, ys_num, color + marker + '--', label=method_label)

    plt.xlabel('x')
    plt.ylabel('y(x)')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()


def main():
    # Параметры
    x_start, x_finish, h = 0.0, 1.0, 0.1
    y_start = [1.0, 0.0]

    xs = generate_grid(x_start, x_finish, h)
    xs_h2 = generate_grid(x_start, x_finish, h / 2.0)

    res_eu = euler_method(ode_system, xs, y_start)
    res_ie1 = improved_euler_first(ode_system, xs, y_start)
    res_ie2 = euler_cauchy_method(ode_system, xs, y_start)
    res_rk = runge_kutta_4(ode_system, xs, y_start)
    res_ad = adams_bashforth_4(ode_system, xs, y_start)

    res_eu_h2 = euler_method(ode_system, xs_h2, y_start)
    res_ie1_h2 = improved_euler_first(ode_system, xs_h2, y_start)
    res_ie2_h2 = euler_cauchy_method(ode_system, xs_h2, y_start)
    res_rk_h2 = runge_kutta_4(ode_system, xs_h2, y_start)
    res_ad_h2 = adams_bashforth_4(ode_system, xs_h2, y_start)

    h1 = (
        f"{'x':^4} | {'РЕШЕНИЯ (y)':^75} || "
        f"{'ОШИБКА РУНГЕ-РОМБЕРГА':^55} | "
        f"{'АБСОЛЮТНАЯ ПОГРЕШНОСТЬ':^55} | "
        f"{'ОТНОСИТЕЛЬНАЯ (%)':^55}"
    )
    h2 = (
        f"{'':^4} | "
        f"{'Точное':^12} {'Эйлер':^12} {'Ул1':^12} {'Эйлер-Коши':^12} {'РК4':^12} {'Адамс':^12} || "
        f"{'Эйлер':^10} {'Ул1':^10} {'Эйл-Коши':^10} {'РК4':^10} {'Адамс':^10} | "
        f"{'Эйлер':^10} {'Ул1':^10} {'Эйл-Коши':^10} {'РК4':^10} {'Адамс':^10} | "
        f"{'Эйлер':^9} {'Ул1':^9} {'Эйл-Коши':^9} {'РК4':^9} {'Адамс':^9}"
    )

    print("=" * len(h1))
    print(h1)
    print("-" * len(h1))
    print(h2)
    print("=" * len(h1))

    for i in range(len(xs)):
        x_curr = xs[i]
        y_ex = exact_solution(x_curr)

        y_eu = res_eu[i][0]
        y_ie1 = res_ie1[i][0]
        y_ie2 = res_ie2[i][0]
        y_rk = res_rk[i][0]
        y_ad = res_ad[i][0]

        # Значения на мелкой сетке (индекс i*2)
        y_eu_small = res_eu_h2[i * 2][0]
        y_ie1_small = res_ie1_h2[i * 2][0]
        y_ie2_small = res_ie2_h2[i * 2][0]
        y_rk_small = res_rk_h2[i * 2][0]
        y_ad_small = res_ad_h2[i * 2][0]

        # Ошибки
        err_eu = calculate_point_errors(y_eu, y_eu_small, y_ex, p=1)
        err_ie1 = calculate_point_errors(y_ie1, y_ie1_small, y_ex, p=2)
        err_ie2 = calculate_point_errors(y_ie2, y_ie2_small, y_ex, p=2)
        err_rk = calculate_point_errors(y_rk, y_rk_small, y_ex, p=4)
        err_ad = calculate_point_errors(y_ad, y_ad_small, y_ex, p=4)

        rr_eu, abs_eu, rel_eu = err_eu
        rr_ie1, abs_ie1, rel_ie1 = err_ie1
        rr_ie2, abs_ie2, rel_ie2 = err_ie2
        rr_rk, abs_rk, rel_rk = err_rk
        rr_ad, abs_ad, rel_ad = err_ad

        print(
            f"{x_curr:^4.1f} | "
            f"{y_ex:^12.8f} {y_eu:^12.8f} {y_ie1:^12.8f} {y_ie2:^12.8f} {y_rk:^12.8f} {y_ad:^12.8f} || "
            f"{rr_eu:^10.2e} {rr_ie1:^10.2e} {rr_ie2:^10.2e} {rr_rk:^10.2e} {rr_ad:^10.2e} | "
            f"{abs_eu:^10.2e} {abs_ie1:^10.2e} {abs_ie2:^10.2e} {abs_rk:^10.2e} {abs_ad:^10.2e} | "
            f"{rel_eu:^9.5f} {rel_ie1:^9.5f} {rel_ie2:^9.5f} {rel_rk:^9.5f} {rel_ad:^9.5f}"
        )

    ys_eu = [y[0] for y in res_eu]
    ys_ie1 = [y[0] for y in res_ie1]
    ys_ie2 = [y[0] for y in res_ie2]
    ys_rk = [y[0] for y in res_rk]
    ys_ad = [y[0] for y in res_ad]

    plot_method(xs, ys_eu, 'Метод Эйлера', 'Эйлер', color='r', marker='o')
    plot_method(xs, ys_ie1, 'Первый улучшенный метод Эйлера (midpoint)',
                'Улучшенный Эйлер 1', color='b', marker='s')
    plot_method(xs, ys_ie2, 'Метод Эйлера–Коши (Heun)',
                'Эйлер–Коши', color='g', marker='^')
    plot_method(xs, ys_rk, 'Метод Рунге–Кутты 4-го порядка',
                'Рунге–Кутта 4', color='m', marker='d')
    plot_method(xs, ys_ad, 'Метод Адамса–Башфорта 4-го порядка',
                'Адамс 4', color='c', marker='x')

    plt.show()

    plt.figure(figsize=(10, 7))

    ys_exact = [exact_solution(x) for x in xs]
    ys_eu = [y[0] for y in res_eu]
    ys_ie1 = [y[0] for y in res_ie1]
    ys_ie2 = [y[0] for y in res_ie2]
    ys_rk = [y[0] for y in res_rk]
    ys_ad = [y[0] for y in res_ad]

    plt.plot(xs, ys_exact, 'k-', linewidth=2, label='Точное решение')

    plt.plot(xs, ys_eu, 'r--o', label='Эйлер ', markersize=5)
    plt.plot(xs, ys_ie1, 'b--s', label='Улучш. Эйлер 1', markersize=5)
    plt.plot(xs, ys_ie2, 'g--^', label='Эйлер–Коши ', markersize=5)
    plt.plot(xs, ys_rk, 'm--d', label='Рунге–Кутта 4 ', markersize=5)
    plt.plot(xs, ys_ad, 'c--x', label='Адамс 4', markersize=5)

    plt.xlabel('x', fontsize=12)
    plt.ylabel('y(x)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.show()
if __name__ == "__main__":
    main()
