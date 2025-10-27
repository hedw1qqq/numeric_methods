import math
import numpy as np
from matplotlib import pyplot as plt

a = 3

x1_vals = np.linspace(-2, 8, 400)
x2_vals = np.linspace(-2, 5, 400)
X1, X2 = np.meshgrid(x1_vals, x2_vals)

F1 = (X1 ** 2 + a ** 2) * X2 - a ** 2
F2 = (X1 - a / 2) ** 2 + (X2 - a / 2) ** 2 - a ** 2

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
    """
    Итерационная функция для метода простой итерации.
    Преобразуем систему к виду x = phi(x).
    """
    x1, x2 = x
    # Из первого уравнения выражаем x2
    x2_new = (a ** 2) / (x1 ** 2 + a ** 2)

    # Из второго уравнения выражаем x1
    discriminant = a ** 2 - (x2_new - a / 2) ** 2
    if discriminant < 0:
        return [x1, x2_new]
    x1_new = a / 2 + math.sqrt(discriminant)
    return [x1_new, x2_new]


def jacobian(x: list[float]) -> list[list[float]]:
    x1, x2 = x
    return [[2 * x1 * x2, x1 ** 2 + a ** 2],
            [2 * (x1 - a / 2), 2 * (x2 - a / 2)]]


def jacobian_phi(x: list[float]) -> list[list[float]]:
    x1, x2 = x

    # Вычисляем промежуточное значение x2 из phi
    x2_phi = (a ** 2) / (x1 ** 2 + a ** 2)
    discriminant = a ** 2 - (x2_phi - a / 2) ** 2

    if discriminant <= 0:
        return [[0, 0], [0, 0]]

    # Производная phi1 по x1: phi1 не зависит от x1, поэтому 0
    dphi1_dx1 = 0

    # Производная phi1 по x2: используем правило производной сложной функции
    dphi1_dx2 = -(x2_phi - a / 2) / math.sqrt(discriminant)

    # Производная phi2 по x1: дифференцируем 9/(x1^2 + 9)
    dphi2_dx1 = -18 * x1 / (x1 ** 2 + 9) ** 2

    # Производная phi2 по x2: phi2 не зависит от x2, поэтому 0
    dphi2_dx2 = 0

    return [[dphi1_dx1, dphi1_dx2],
            [dphi2_dx1, dphi2_dx2]]


def matrix_norm(M: list[list[float]]) -> float:
    """ Норма матрицы: максимум суммы модулей элементов по строкам"""
    return max(sum(abs(M[i][j]) for j in range(len(M[0]))) for i in range(len(M)))


def norm(v: list[float]) -> float:
    return math.sqrt(sum(x ** 2 for x in v))


def solve_system(A: list[list[float]], b: list[float]) -> list[float]:
    """Решение СЛАУ 2x2 методом Крамера"""
    det = A[0][0] * A[1][1] - A[0][1] * A[1][0]
    if abs(det) < 1e-18:
        raise ValueError("Матрица вырождена")
    x1 = (b[0] * A[1][1] - b[1] * A[0][1]) / det
    x2 = (A[0][0] * b[1] - A[1][0] * b[0]) / det
    return [x1, x2]


def check_newton_convergence(f, jacobian, x0):
    fx0 = f(x0)
    J = jacobian(x0)
    det = J[0][0] * J[1][1] - J[0][1] * J[1][0]

    print("\n=== Проверка условий сходимости метода Ньютона ===")
    print(f"F(x0) = [{fx0[0]:.6f}, {fx0[1]:.6f}]")
    print(f"det(J(x0)) = {det:.6f}")

    if abs(det) < 1e-10:
        print("Определитель якобиана близок к нулю - метод может не работать")
        return False
    else:
        print("Якобиан невырожден - условие выполнено")
        return True


def check_simple_iteration_convergence(phi, jacobian_phi, x0):
    J_phi = jacobian_phi(x0)
    q = matrix_norm(J_phi)

    print("\n=== Проверка условий сходимости метода простой итерации ===")
    print(f"Якобиан phi:")
    print(f"  [[{J_phi[0][0]:.6f}, {J_phi[0][1]:.6f}],")
    print(f"   [{J_phi[1][0]:.6f}, {J_phi[1][1]:.6f}]]")
    print(f"Норма якобиана phi: ||J_phi(x0)|| = {q:.6f}")

    if q >= 1:
        print(f"Условие сходимости НЕ выполнено: ||J_phi|| = {q:.6f} >= 1")
        return False, q
    else:
        print(f"Условие сходимости выполнено: ||J_phi|| = {q:.6f} < 1")
        return True, q


def simple_iteration(x0: list[float], tol: float) -> tuple[list[float], list[int], list[float]]:
    """
    Метод простой итерации: x(k+1) = phi(x(k))
    Останавливается когда ||x(k+1) - x(k)|| < tol
    """
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
    """
    Метод Ньютона: x(k+1) = x(k) - J^(-1) * F(x(k))
    На каждой итерации решаем линейную систему J*delta = -F(x)
    """
    x = x0.copy()
    iterations = []
    errors = []

    for k in range(100):
        fx = f(x)
        J = jacobian(x)
        #  J * delta = -F(x) методом Крамера
        delta = solve_system(J, [-fx[0], -fx[1]])

        x_new = [x[i] + delta[i] for i in range(2)]
        error = norm(delta)

        iterations.append(k + 1)
        errors.append(error)

        if error < tol:
            return x_new, iterations, errors
        x = x_new

    return x, iterations, errors


x0 = [3, 0.5]  # Определено графически
tol = 1e-12

print("=" * 60)
print("РЕШЕНИЕ СИСТЕМЫ НЕЛИНЕЙНЫХ УРАВНЕНИЙ")
print("=" * 60)
print("\nСистема уравнений:")
print(f"(x₁² + 9)x₂ - 9 = 0")
print(f"(x₁ - 1.5)² + (x₂ - 1.5)² - 9 = 0\n")

print(f"Начальное приближение: x0 = {x0}")
print(f"Точность: {tol}")

converges_si, q = check_simple_iteration_convergence(phi, jacobian_phi, x0)
converges_n = check_newton_convergence(f, jacobian, x0)

print("\n" + "=" * 60)
print("МЕТОД ПРОСТОЙ ИТЕРАЦИИ")
print("=" * 60)
sol_si, iter_si, err_si = simple_iteration(x0, tol)
print(f"\nРешение: x1 = {sol_si[0]:.8f}, x2 = {sol_si[1]:.8f}")
print(f"Итераций: {len(iter_si)}")
print(f"Положительные: {all(xi > 0 for xi in sol_si)}")
print(f"Коэффициент сжатия q = {q:.6f}")

print("\n" + "=" * 60)
print("МЕТОД НЬЮТОНА")
print("=" * 60)
sol_n, iter_n, err_n = newton(x0, tol)
print(f"\nРешение: x1 = {sol_n[0]:.8f}, x2 = {sol_n[1]:.8f}")
print(f"Итераций: {len(iter_n)}")
print(f"Положительные: {all(xi > 0 for xi in sol_n)}")

print("\n" + "=" * 60)
print("АНАЛИЗ ПОГРЕШНОСТИ ОТ ИТЕРАЦИЙ")
print("=" * 60)
print("\nПростая итерация:")
for i in [0, 1, 2, -3, -2, -1]:
    if abs(i) < len(iter_si):
        print(f"  Итерация {iter_si[i]}: погрешность = {err_si[i]:.2e}")

print("\nМетод Ньютона:")
for i in range(len(iter_n)):
    print(f"  Итерация {iter_n[i]}: погрешность = {err_n[i]:.2e}")

print("\n" + "=" * 60)
print("СРАВНЕНИЕ МЕТОДОВ")
print("=" * 60)
print(f"Метод простой итерации: {len(iter_si)} итераций (q={q:.4f})")
print(f"Метод Ньютона: {len(iter_n)} итераций")
print(f"Ньютон быстрее в {len(iter_si) / len(iter_n):.1f} раз")
