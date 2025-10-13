from typing import Callable, List, Tuple
import math


def fixed_point_iteration(
        phi: Callable[[float], float],
        x0: float,
        eps: float,
        max_iter: int = 100
) -> Tuple[float, List[Tuple[int, float, float]]]:
    iterations = []
    x = x0

    for i in range(max_iter):
        x_new = phi(x)
        error = abs(x_new - x)
        iterations.append((i, x_new, error))

        if error < eps:
            return x_new, iterations
        x = x_new

    return x, iterations


def newton_method(
        f: Callable[[float], float],
        df: Callable[[float], float],
        x0: float,
        eps: float,
        max_iter: int = 100
) -> Tuple[float, List[Tuple[int, float, float]]]:
    iterations = []
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-10:
            print("Производная близка к нулю")
            break

        x_new = x - fx / dfx
        error = abs(x_new - x)
        iterations.append((i, x_new, error))

        if error < eps:
            return x_new, iterations
        x = x_new

    return x, iterations


#  ln(x+2) -x^2 = 0
def f(x: float) -> float:
    return math.log(x + 2) - x ** 2


def df(x: float) -> float:
    return 1.0 / (x + 2) - 2 * x


def phi(x: float) -> float:
    return math.sqrt(math.log(x + 2))


eps = 1e-12
x0 = 1.5

print("=== Метод простой итерации ===")
root1, iter1 = fixed_point_iteration(phi, x0, eps)
print(f"Корень: {root1:.10f}")
print(f"Итераций: {len(iter1)}")
print(f"f(x) = {f(root1):.2e}\n")

print("Итерация | Приближение | Погрешность")
print("-" * 45)
for i, x, err in iter1:
    print(f"{i:8d} | {x:11.8f} | {err:.2e}")

print("\n=== Метод Ньютона ===")
root2, iter2 = newton_method(f, df, x0, eps)
print(f"Корень: {root2:.10f}")
print(f"Итераций: {len(iter2)}")
print(f"f(x) = {f(root2):.2e}\n")

print("Итерация | Приближение | Погрешность")
print("-" * 45)
for i, x, err in iter2:
    print(f"{i:8d} | {x:11.8f} | {err:.2e}")

print("\n=== Анализ сходимости ===")
print(f"Метод простой итерации: {len(iter1)} итераций")
print(f"Метод Ньютона: {len(iter2)} итераций")
