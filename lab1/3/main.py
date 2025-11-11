from utility.Matrix import Matrix
import numpy as np


def check_diagonal_dominance(A: "Matrix"):
    is_dominant = True
    for i in range(A.n):
        diag_element = abs(A.data[i][i])
        sum_of_others = sum(abs(A.data[i][j]) for j in range(A.n) if i != j)
        if diag_element <= sum_of_others:
            is_dominant = False
            break
    if not is_dominant:
        print("Матрица не обладает строгим диагональным преобладанием.")


def solve_iterative(A: "Matrix", b: list[float], eps: float = 1e-6, max_iter: int = 1000) -> list[float]:
    n = A.n
    if n != len(b):
        raise ValueError("Размер матрицы и вектора должен совпадать.")

    check_diagonal_dominance(A)

    x = [0.0] * n
    B = Matrix.zeros(n)
    c = [0.0] * n

    for i in range(n):
        if abs(A.data[i][i]) < 1e-15:
            raise ZeroDivisionError(f"Нулевой или слишком маленький диагональный элемент в строке {i}.")
        for j in range(n):
            if i != j:
                B.data[i][j] = -A.data[i][j] / A.data[i][i]
        c[i] = b[i] / A.data[i][i]

    alpha = max(sum(abs(val) for val in row) for row in B.data)
    print(f"Норма матрицы итераций: {alpha:.5f}")

    for it in range(1, max_iter + 1):
        x_new = [sum(B.data[i][j] * x[j] for j in range(n)) + c[i] for i in range(n)]

        # ||x^(k) - x^(k-1)||_inf
        diff_norm = max(abs(x_new[i] - x[i]) for i in range(n))

        error_estimate = (alpha / (1 - alpha)) * diff_norm

        if error_estimate < eps:
            print(f"Решение методом итераций достигнуто за {it} итераций")
            return x_new

        x = x_new

    raise ValueError(f"Решение не найдено за {max_iter} итераций")


def solve_seidel(A: Matrix, b: list[float], eps: float = 1e-6, max_iter: int = 1000000) -> list[float]:
    n = A.n
    if n != len(b):
        raise ValueError("Размер матрицы и вектора должен совпадать.")

    check_diagonal_dominance(A)

    # A = D + L + U
    L = Matrix.zeros(n)
    D = Matrix.zeros(n)
    U = Matrix.zeros(n)
    # B_Seidel = -(L + D) ^ (-1) * U

    for i in range(n):
        D.data[i][i] = A.data[i][i]
        for j in range(n):
            if j < i:
                L.data[i][j] = A.data[i][j]
            elif j > i:
                U.data[i][j] = A.data[i][j]

    L_plus_D = L + D

    L_plus_D_inv = Matrix.inverse(L_plus_D)

    B_seidel = L_plus_D_inv @ U
    B_seidel = -B_seidel

    alpha = max(sum(abs(val) for val in row) for row in B_seidel.data)
    print(f"Норма матрицы итераций Зейделя: {alpha:.5f}")

    x = [0.0] * n
    # xi^k+1 = 1/a[i][i] ( b[i]- sum(a[i][j] * x[j]^k+1) (j < i) - sum(a[i][j]x^k[j])
    for k in range(1, max_iter + 1):
        x_old = x[:]

        for i in range(n):
            if abs(A.data[i][i]) < 1e-15:
                raise ZeroDivisionError(f"Нулевой диагональный элемент в строке {i}.")

            sum1 = sum(A.data[i][j] * x[j] for j in range(i))
            sum2 = sum(A.data[i][j] * x_old[j] for j in range(i + 1, n))
            x[i] = (b[i] - sum1 - sum2) / A.data[i][i]

        diff_norm = max(abs(x[i] - x_old[i]) for i in range(n))

        if alpha < 1:
            error_estimate = (alpha / (1 - alpha)) * diff_norm
            if error_estimate < eps:
                print(f"Решение методом Зейделя достигнуто за {k} итераций")
                return x
        else:
            if diff_norm < eps:
                print(f"Решение методом Зейделя достигнуто за {k} итераций")
                return x

    raise ValueError(f"Решение не найдено за {max_iter} итераций")


if __name__ == "__main__":
    A, b = Matrix.read_system_from_file("input")

    try:
        x_mysol = solve_iterative(A, b)
        print(f"ITERATIVE METHOD SOLUTION\nMY: {x_mysol}")

        A_np = np.array(A.data)
        b_np = np.array(b)
        x_npsol = np.linalg.solve(A_np, b_np)
        print(f"NUMPY: {x_npsol}")

    except (ValueError, ZeroDivisionError) as e:
        print(f"Ошибка в методе итераций: {e}")

    print("-" * 40)

    try:
        x_zeid = solve_seidel(A, b)
        print(f"SEIDEL METHOD SOLUTION\nMY: {x_zeid}")

        A_np = np.array(A.data)
        b_np = np.array(b)
        x_npsol = np.linalg.solve(A_np, b_np)
        print(f"NUMPY: {x_npsol}")

    except (ValueError, ZeroDivisionError) as e:
        print(f"Ошибка в методе Зейделя: {e}")
