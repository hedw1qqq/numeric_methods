import math
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable, List, Tuple


def f(x: float) -> float:
    return math.log(x + 2) - x ** 2


def df(x: float) -> float:
    return 1.0 / (x + 2) - 2 * x


def d2f(x: float) -> float:
    return -1.0 / (x + 2) ** 2 - 2


def phi(x: float) -> float:
    return math.sqrt(math.log(x + 2))


def phi_derivative(x: float) -> float:
    return 1.0 / (2 * (x + 2) * math.sqrt(math.log(x + 2)))


xs = np.linspace(-0.9, 5, 1000)
fxs = [math.log(x + 2) - x ** 2 for x in xs]

plt.figure(figsize=(8, 4))
plt.plot(xs, fxs, label='f(x) = ln(x+2) - x^2')
plt.axhline(0, color='k', linewidth=0.5)
plt.scatter([1.5], [0], color='red', label='root (approx)')
plt.title('График f(x) — для выбора начального приближения')
plt.legend()
plt.grid(True)
plt.savefig('system_plot.png')
plt.show()


def fixed_point_iteration(phi, phi_der, x0, eps, a, b, max_iter=100):
    print(f"\n=== Проверка условий сходимости метода простой итерации ===")

    xs_check = np.linspace(a, b, 1000)
    q = max(abs(phi_der(x)) for x in xs_check)
    print(f"q = max|phi'(x)| на [{a}, {b}] = {q:.6f}")

    if q >= 1:
        raise ValueError(f"Условие сходимости НЕ выполнено: q = {q:.6f} >= 1")
    print(f"Условие сходимости выполнено: q < 1")

    iterations = []
    x = x0

    for i in range(max_iter):
        x_new = phi(x)

        if not (a <= x_new <= b):
            print(f"Предупреждение: phi({x:.4f}) = {x_new:.4f} вне [{a}, {b}]")

        error = abs(x_new - x)
        iterations.append((i, x_new, error))

        if error < eps:
            return x_new, iterations, q
        x = x_new

    raise ValueError(f"Не сошлось за {max_iter} итераций")


def newton_method(f, df, d2f, x0, eps, max_iter=100):
    print(f"\n=== Проверка условий сходимости метода Ньютона ===")

    fx0 = f(x0)
    d2fx0 = d2f(x0)
    product = fx0 * d2fx0

    print(f"f(x0) = {fx0:.6f}")
    print(f"f''(x0) = {d2fx0:.6f}")
    print(f"f(x0) * f''(x0) = {product:.6f}")

    if product <= 0:
        raise ValueError(f"Условие сходимости НЕ выполнено: f(x0)*f''(x0) = {product:.6f} <= 0")
    print(f"Условие сходимости выполнено: f(x0)*f''(x0) > 0")

    iterations = []
    x = x0

    for i in range(max_iter):
        fx = f(x)
        dfx = df(x)

        if abs(dfx) < 1e-10:
            raise ValueError("Производная близка к нулю")

        x_new = x - fx / dfx
        error = abs(x_new - x)
        iterations.append((i, x_new, error))

        if error < eps:
            return x_new, iterations
        x = x_new

    raise ValueError(f"Не сошлось за {max_iter} итераций")


eps = 1e-12
x0 = 1.5
a, b = 0.5, 2.0

print("=" * 60)
print("РЕШЕНИЕ УРАВНЕНИЯ ln(x+2) - x² = 0")
print("=" * 60)

print("\n" + "=" * 60)
print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
print("=" * 60)
root1, iter1, q = fixed_point_iteration(phi, phi_derivative, x0, eps, a, b)
print(f"\nКорень: {root1:.10f}")
print(f"Итераций: {len(iter1)}")
print(f"f(x) = {f(root1):.2e}")
print(f"Коэффициент сжатия q = {q:.6f}")

print("\nИтерация | Приближение | Погрешность")
print("-" * 45)
for i, x, err in iter1[:5]:
    print(f"{i:8d} | {x:11.8f} | {err:.2e}")
if len(iter1) > 5:
    print("...")
    for i, x, err in iter1[-2:]:
        print(f"{i:8d} | {x:11.8f} | {err:.2e}")

print("\n" + "=" * 60)
print("МЕТОД НЬЮТОНА")
print("=" * 60)
root2, iter2 = newton_method(f, df, d2f, x0, eps)
print(f"\nКорень: {root2:.10f}")
print(f"Итераций: {len(iter2)}")
print(f"f(x) = {f(root2):.2e}")

print("\nИтерация | Приближение | Погрешность")
print("-" * 45)
for i, x, err in iter2:
    print(f"{i:8d} | {x:11.8f} | {err:.2e}")

print("\n" + "=" * 60)
print("АНАЛИЗ СХОДИМОСТИ")
print("=" * 60)
print(f"Метод простой итерации: {len(iter1)} итераций (линейная сходимость, q={q:.4f})")
print(f"Метод Ньютона: {len(iter2)} итераций (квадратичная сходимость)")
print(f"Ньютон быстрее в {len(iter1) / len(iter2):.1f} раз")
