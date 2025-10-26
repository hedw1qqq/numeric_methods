import math
import numpy as np
from matplotlib import pyplot as plt
a = 3
x1_vals = np.linspace(-2, 8, 400)
x2_vals = np.linspace(-2, 5, 400)

X1, X2 = np.meshgrid(x1_vals, x2_vals)

# Первое уравнение: (x1^2 + 9)*x2 - 9 = 0
F1 = (X1**2 + a**2) * X2 - a**2

# Второе уравнение: (x1 - 1.5)^2 + (x2 - 1.5)^2 - 9 = 0
F2 = (X1 - a/2)**2 + (X2 - a/2)**2 - a**2

plt.figure(figsize=(10, 8))
plt.contour(X1, X2, F1, levels=[0], colors='blue', label='F1=0')
plt.contour(X1, X2, F2, levels=[0], colors='red', label='F2=0')
plt.scatter([3], [0.5], color='green', s=100, label='Приближенное решение', zorder=5)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title('Графическое определение начального приближения')
plt.legend()
plt.grid(True)
plt.savefig('system_plot.png')
plt.show()

def f(x: list[float]) -> list[float]:
    x1, x2 = x
    f1 = (x1 ** 2 + a ** 2) * x2 - a ** 2
    f2 = (x1 - a / 2) ** 2 + (x2 - a / 2) ** 2 - a ** 2
    return [f1, f2]


def phi(x: list[float]) -> list[float]:
    x1, x2 = x
    x2_new = (a ** 2) / (x1 ** 2 + a ** 2)
    discriminant = a ** 2 - (x2_new - a / 2) ** 2
    if discriminant < 0:
        return [x1, x2_new]
    x1_new = a / 2 + math.sqrt(discriminant)
    return [x1_new, x2_new]


def jacobian(x: list[float]) -> list[list[float]]:
    x1, x2 = x
    return [[2 * x1 * x2, x1 ** 2 + a ** 2],
            [2 * (x1 - a / 2), 2 * (x2 - a / 2)]]


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in v))


def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-18:
        raise ValueError("Матрица вырождена")
    x1 = (b[0] * A[1][1] - b[1] * A[0][1]) / det
    x2 = (A[0][0] * b[1] - A[1][0] * b[0]) / det
    return [x1, x2]


def simple_iteration(x0: list[float], tol: float) -> tuple[list[float], list[int], list[float]]:
    x = x0.copy()
    iterations = []
    errors = []

    for k in range(10000):
        x_new = phi(x)
        error = norm([x_new[i] - x[i] for i in range(2)])

        iterations.append(k + 1)
        errors.append(error)

        if error < tol:
            return x_new, iterations, errors
        x = x_new

    return x, iterations, errors


def newton(x0: list[float], tol: float) -> tuple[list[float], list[int], list[float]]:
    x = x0.copy()
    iterations = []
    errors = []

    for k in range(100):
        fx = f(x)
        J = jacobian(x)
        delta = solve_system(J, [-fx[0], -fx[1]])

        x_new = [x[i] + delta[i] for i in range(2)]
        error = norm(delta)

        iterations.append(k + 1)
        errors.append(error)

        if error < tol:
            return x_new, iterations, errors
        x = x_new

    return x, iterations, errors


x0 = [3, 0.5]  # почти решение - [4.26, 0.33]
tol = 1e-12

print("Система уравнений:")
print(f"(x₁² + 9)x₂ - 9 = 0")
print(f"(x₁ - 1.5)² + (x₂ - 1.5)² - 9 = 0\n")

print(f"Начальное приближение: x0 = {x0}")
print(f"Точность: {tol}\n")

print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
sol_si, iter_si, err_si = simple_iteration(x0, tol)
print(f"Решение: x1 = {sol_si[0]:.8f}, x2 = {sol_si[1]:.8f}")
print(f"Итераций: {len(iter_si)}")
print(f"Положительные: {all(xi > 0 for xi in sol_si)}\n")

print("МЕТОД НЬЮТОНА")
sol_n, iter_n, err_n = newton(x0, tol)
print(f"Решение: x1 = {sol_n[0]:.8f}, x2 = {sol_n[1]:.8f}")
print(f"Итераций: {len(iter_n)}")
print(f"Положительные: {all(xi > 0 for xi in sol_n)}\n")

print("АНАЛИЗ ПОГРЕШНОСТИ ОТ ИТЕРАЦИЙ:")
print("\nПростая итерация:")
for i in [0, 1, 2, -3, -2, -1]:
    if abs(i) < len(iter_si):
        print(f"  Итерация {iter_si[i]}: погрешность = {err_si[i]:.2e}")

print("\nМетод Ньютона:")
for i in range(len(iter_n)):
    print(f"  Итерация {iter_n[i]}: погрешность = {err_n[i]:.2e}")
