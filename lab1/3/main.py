from utility.Matrix import Matrix
import numpy as np


def solve_iterative(A: "Matrix", b: list[float], eps: float = 1e-6, max_iter: int = 1000) -> list[float]:
    n = A.n
    if n != len(b):
        raise ValueError("Размер матрицы и вектора должен совпадать.")
    x = [0.0] * n  # начальное приближение
    B = Matrix.zeros(n)  # матрица итераций
    c = [0.0] * n
    for i in range(n):
        for j in range(n):
            if abs(A.data[i][i]) < 1e-15:
                raise ZeroDivisionError
            if i != j:
                B.data[i][j] = -A.data[i][j] / A.data[i][i]
        c[i] = b[i] / A.data[i][i]

    for i in range(1, max_iter + 1):
        x_new = [sum(B.data[i][j] * x[j] for j in range(n)) + c[i] for i in range(n)]
        if max(abs(x_new[i] - x[i]) for i in range(n)) < eps:
            print(f"Решение достигнуто за {max_iter} итераций")
            return x_new
        x = x_new
    raise ValueError(f"Решение не найдено за {max_iter} итераций")


def solve_seidel(A: Matrix, b: list[float], eps: float = 1e-6, max_iter: int = 1000000) -> list[float]:
    n = A.n
    if n != len(b):
        raise ValueError("Размер матрицы и вектора должен совпадать.")

    # начальное приближение
    x = [0.0] * n

    # итерационный процесс
    for k in range(1, max_iter + 1):
        x_new = x[:]  # копия текущего приближения

        for i in range(n):
            if abs(A.data[i][i]) < 1e-15:
                raise ZeroDivisionError

            # сумма с уже обновлёнными компонентами (j < i)
            sum1 = sum(A.data[i][j] * x_new[j] for j in range(i))
            # сумма с "старыми" компонентами (j > i)
            sum2 = sum(A.data[i][j] * x[j] for j in range(i + 1, n))

            x_new[i] = (b[i] - sum1 - sum2) / A.data[i][i]

        diff = max(abs(x_new[i] - x[i]) for i in range(n))
        if diff < eps:
            print(f"Решение достигнуто за {k} итераций")
            return x_new

        x = x_new

    raise ValueError(f"Решение не найдено за {max_iter} итераций")


if __name__ == "__main__":
    A, b = Matrix.read_system_from_file("input")
    A_np = np.array(A.data)
    b_np = np.array(b)
    x_mysol = solve_iterative(A, b)
    x_npsol = np.linalg.solve(A_np, b_np)
    print(f"ITERATIVE METHOD SOLUTION\nMY\n{x_mysol}\nNUMPY \n{x_npsol}")
    print("-" * 40)
    x_zeid = solve_seidel(A, b)

    print(f"SEIDEL METHOD SOLUTION\n{x_zeid}")
