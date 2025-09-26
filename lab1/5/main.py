from utility.Matrix import Matrix
import math
import numpy as np


def qr_decomposition(A: "Matrix") -> tuple["Matrix", "Matrix"]:
    n = A.n
    Q = Matrix.zeros(n)
    R = Matrix.zeros(n)

    for j in range(n):
        v = A._get_col(j)
        a_j = v[:]

        for i in range(j):
            q_i = Q._get_col(i)

            # r_ij = q_i^T * a_j
            r_ij = Matrix._dot(q_i, a_j)
            R.data[i][j] = r_ij

            # v = v - r_ij * q_i
            v = [v_k - r_ij * q_i_k for v_k, q_i_k in zip(v, q_i)]

        norm_v = Matrix._norm(v)

        if norm_v < 1e-12:
            raise ValueError("Матрица вырождена")

        R.data[j][j] = norm_v

        # q_j = v / ||v||
        q_j = [v_k / norm_v for v_k in v]
        Q._set_col(j, q_j)

    return Q, R


def find_eigenvalues_qr(A: "Matrix", epsilon: float = 1e-12, max_iterations: int = 1000) -> list[float]:
    Ak = A.copy()
    n = A.n

    for i in range(max_iterations):
        try:
            Q, R = qr_decomposition(Ak)
        except ValueError as e:
            print(f"Ошибка QR-разложения на итерации {i}: {e}")
            break

        Ak = R @ Q

        if Ak._off_diagonal_sum() < epsilon:
            print(f"Сходимость достигнута на итерации {i}.")
            break
    else:
        print(f"Превышено максимальное количество итераций ({max_iterations}).")

    eigenvalues = [Ak.data[i][i] for i in range(n)]
    return eigenvalues


if __name__ == "__main__":
    A = Matrix.read_from_file("input")
    A_np = np.array(A.data)
    eigenvalues = find_eigenvalues_qr(A)
    eigenvalues_np = np.linalg.eig(A_np)
    print(f"MY\n{eigenvalues}\nNP\n{eigenvalues_np}")
