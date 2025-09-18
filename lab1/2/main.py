from utility.Matrix import Matrix
import numpy as np


def solve_progonka(A, d: list[float]) -> list[float]:
    n = A.n
    if n != len(d):
        raise ValueError("Размер матрицы и вектора должен совпадать.")

    for i in range(n):
        for j in range(n):
            if abs(i - j) > 1 and abs(A.data[i][j]) > 1e-12:
                raise ValueError(f"Матрица не является трёхдиагональной")

    b = [A.data[i][i] for i in range(n)]
    a = [A.data[i + 1][i] for i in range(n - 1)]
    c = [A.data[i][i + 1] for i in range(n - 1)]

    alpha = [0.0] * n
    beta = [0.0] * n

    if abs(b[0]) < 1e-12:
        raise ValueError("Нулевой элемент на главной диагонали (b[0])")
    alpha[0] = -c[0] / b[0]
    beta[0] = d[0] / b[0]

    for i in range(1, n):
        denom = b[i] + a[i - 1] * alpha[i - 1]
        if abs(denom) < 1e-12:
            raise ValueError(f"Нулевой знаменатель на шаге {i}")
        if i < n - 1:
            alpha[i] = -c[i] / denom
        beta[i] = (d[i] - a[i - 1] * beta[i - 1]) / denom

    x = [0.0] * n
    x[n - 1] = beta[n - 1]
    for i in range(n - 2, -1, -1):
        x[i] = alpha[i] * x[i + 1] + beta[i]

    return x


if __name__ == "__main__":
    A, b = Matrix.read_system_from_file("input")
    A_np = np.array(A.data)
    b_np = np.array(b)
    print(solve_progonka(A, b))
    print("-" * 40)
    print(np.linalg.solve(A_np, b_np))
