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
    return math.exp(-x) / x


def func_shoot(x: float, Y: Vector) -> Vector:
    """
    Система для метода стрельбы:
    y1' = y2
    y2' = y1 - (2/x)*y2
    """
    y1, y2 = Y
    return [y2, y1 - (2.0 / x) * y2]


# --- 1. Метод стрельбы ---

def rk4_step(f: Callable, x: float, Y: Vector, h: float) -> Vector:
    k1 = vec_scale(f(x, Y), h)
    y_k2 = vec_add(Y, vec_scale(k1, 0.5))
    k2 = vec_scale(f(x + 0.5 * h, y_k2), h)
    y_k3 = vec_add(Y, vec_scale(k2, 0.5))
    k3 = vec_scale(f(x + 0.5 * h, y_k3), h)
    y_k4 = vec_add(Y, k3)
    k4 = vec_scale(f(x + h, y_k4), h)

    sum_k = vec_add(vec_add(k1, vec_scale(k2, 2.0)), vec_add(vec_scale(k3, 2.0), k4))
    return vec_add(Y, vec_scale(sum_k, 1.0 / 6.0))


def solve_cauchy_on_grid(f: Callable, xs: List[float], Y0: Vector) -> List[Vector]:
    """Решает задачу Коши на переданной сетке xs."""
    ys = [Y0]
    curr_y = Y0

    for i in range(len(xs) - 1):
        x_curr = xs[i]
        x_next = xs[i + 1]
        h = x_next - x_curr  # Вычисляем локальный шаг

        curr_y = rk4_step(f, x_curr, curr_y, h)
        ys.append(curr_y)
    return ys


def shooting_method(
        xs: List[float], y_start: float, y_end: float, eps: float = 1e-6
) -> List[float]:
    """
    Метод стрельбы на заданной сетке xs.
    """
    alpha0, alpha1 = -0.1, -1.0

    # 1. Первый выстрел
    res0 = solve_cauchy_on_grid(func_shoot, xs, [y_start, alpha0])
    phi0 = res0[-1][0] - y_end

    # 2. Второй выстрел
    res1 = solve_cauchy_on_grid(func_shoot, xs, [y_start, alpha1])
    phi1 = res1[-1][0] - y_end

    # 3. Метод секущих
    iter_count = 0
    while abs(phi1) > eps and iter_count < 100:
        denom = phi1 - phi0
        if abs(denom) < 1e-14:
            break

        alpha_next = alpha1 - phi1 * (alpha1 - alpha0) / denom
        alpha0, phi0 = alpha1, phi1
        alpha1 = alpha_next

        res_new = solve_cauchy_on_grid(func_shoot, xs, [y_start, alpha1])
        phi1 = res_new[-1][0] - y_end
        iter_count += 1

    # Финальный расчет
    final_res = solve_cauchy_on_grid(func_shoot, xs, [y_start, alpha1])
    return [v[0] for v in final_res]


# --- 2. Конечно-разностный метод ---

def finite_difference_method(
        xs: List[float], y_start: float, y_end: float
) -> List[float]:
    N = len(xs) - 1
    h = xs[1] - xs[0]

    # y_i = alpha_{i+1} * y_{i+1} + beta_{i+1}
    alpha = [0.0] * N
    beta = [0.0] * N
    alpha[0] = 0.0
    beta[0] = y_start

    # Прямой ход
    for i in range(0, N - 1):

        xi = xs[i + 1]

        # y'' + (2/x)y' - y = 0
        p_val = 2.0 / xi
        q_val = -1.0
        f_val = 0.0

        # Коэффициенты разностной схемы (A y_{i-1} - C y_i + B y_{i+1} = -F)
        # A = 1 - p*h/2
        # B = 1 + p*h/2
        # C = 2 - q*h^2
        # F = f*h^2

        A = 1.0 - p_val * h / 2.0
        B = 1.0 + p_val * h / 2.0
        C = 2.0 - q_val * h * h
        F = f_val * h * h

        # Формулы для alpha и beta на следующем шаге
        # alpha_{i+1} = B / (C - A * alpha_i)
        # beta_{i+1} = (F + A * beta_i) / (C - A * alpha_i)

        denom = C - A * alpha[i]
        alpha[i + 1] = B / denom
        beta[i + 1] = (F + A * beta[i]) / denom

    ys = [0.0] * (N + 1)
    ys[N] = y_end

    for i in range(N - 1, -1, -1):
        ys[i] = alpha[i] * ys[i + 1] + beta[i]

    return ys


# --- Расчет ошибок ---
def calculate_point_errors(
        val_h: float, val_h2: float, val_ex: float, p: int
) -> ErrorTuple:
    rr = abs(val_h - val_h2) / (2 ** p - 1)
    abs_err = abs(val_h - val_ex)
    rel_err = (abs_err / abs(val_ex)) * 100.0 if abs(val_ex) > 1e-14 else 0.0
    return rr, abs_err, rel_err


def main():
    x0, xN = 1.0, 2.0
    y_start = math.exp(-1.0) / 1.0
    y_end = math.exp(-2.0) / 2.0
    h = 0.1

    # 1. Генерация сеток
    xs = generate_grid(x0, xN, h)
    xs_h2 = generate_grid(x0, xN, h / 2.0)

    # 2. Решения на основной сетке
    ys_shoot = shooting_method(xs, y_start, y_end)
    ys_fd = finite_difference_method(xs, y_start, y_end)

    # 3. Решения на мелкой сетке (для оценки точности)
    ys_shoot_h2 = shooting_method(xs_h2, y_start, y_end)
    ys_fd_h2 = finite_difference_method(xs_h2, y_start, y_end)

    # 4. Вывод результатов
    h1 = (
        f"{'x':^4} | {'РЕШЕНИЯ (y)':^38} || "
        f"{'ОШИБКА РУНГЕ-РОМБЕРГА':^21} | "
        f"{'АБСОЛЮТНАЯ ПОГРЕШНОСТЬ':^18} | "
        f"{'ОТНОСИТЕЛЬНАЯ (%)':^21}"
    )

    h2 = (
        f"{'':^4} | {'Точное':^12} {'Стрельба':^12} {'КРМ':^12} || "
        f"{'Стрельба':^10} {'КРМ':^10} | "
        f"{'Стрельба':^10} {'КРМ':^10} | "
        f"{'Стрельба':^9} {'КРМ':^9}"
    )

    line_len = len(h1)

    print("=" * line_len)
    print(h1)
    print("-" * line_len)
    print(h2)
    print("=" * line_len)

    for i in range(len(xs)):
        x = xs[i]
        y_ex = exact_solution(x)

        val_sh = ys_shoot[i]
        val_fd = ys_fd[i]

        # Сравниваем с точками на мелкой сетке
        val_sh_small = ys_shoot_h2[i * 2]
        val_fd_small = ys_fd_h2[i * 2]

        # Ошибки
        # Стрельба (p=4)
        rr_sh, abs_sh, rel_sh = calculate_point_errors(val_sh, val_sh_small, y_ex, p=4)
        # КРМ (p=2)
        rr_fd, abs_fd, rel_fd = calculate_point_errors(val_fd, val_fd_small, y_ex, p=2)

        print(
            f"{x:^4.1f} | "
            f"{y_ex:^12.8f} {val_sh:^12.8f} {val_fd:^12.8f} || "
            f"{rr_sh:^10.2e} {rr_fd:^10.2e} | "
            f"{abs_sh:^10.2e} {abs_fd:^10.2e} | "
            f"{rel_sh:^9.5f} {rel_fd:^9.5f}"
        )


if __name__ == "__main__":
    main()
