import math


class Matrix:
    def __init__(self, data: list[list[float]], n: int) -> None:
        self.data = [list(row) for row in data]
        self.n = n

    def __str__(self) -> str:
        rows = ["[" + " ".join(f"{x:8g}" for x in row) + "]" for row in self.data]
        return "Matrix " + str(self.n) + ":\n" + "\n".join(rows)

    def __len__(self) -> int:
        return self.n

    def copy(self) -> "Matrix":
        return Matrix([row[:] for row in self.data], self.n)

    def __add__(self, other: "Matrix") -> "Matrix":
        self._check_same_size(other)
        return Matrix([[self.data[i][j] + other.data[i][j] for j in range(self.n)] for i in range(self.n)], self.n)

    def __sub__(self, other: "Matrix") -> "Matrix":
        self._check_same_size(other)
        return Matrix([[self.data[i][j] - other.data[i][j] for j in range(self.n)] for i in range(self.n)], self.n)

    def __neg__(self) -> "Matrix":
        return Matrix([[-self.data[i][j] for j in range(self.n)] for i in range(self.n)], self.n)

    def __mul__(self, alpha: float) -> "Matrix":
        # A * c
        return Matrix([[self.data[i][j] * alpha for j in range(self.n)] for i in range(self.n)], self.n)

    def __rmul__(self, alpha: float) -> "Matrix":
        # c * A
        return self.__mul__(alpha)

    def __matmul__(self, other: "Matrix") -> "Matrix":
        # A @ B
        self._check_same_size(other)
        n = self.n
        C = Matrix.zeros(n)

        for i in range(n):
            for k in range(n):
                aik = self.data[i][k]
                if aik == 0.0:
                    continue
                for j in range(n):
                    C.data[i][j] += aik * other.data[k][j]
        return C

    @staticmethod
    def read_from_file(file_path: str) -> "Matrix":
        with open(file_path, 'r') as file:
            lines = file.readlines()
            n = int(lines[0].strip())
            data = [list(map(float, line.strip().split())) for line in lines[1:n + 1]]
            return Matrix(data, n)

    def read_system_from_file(file_path: str) -> tuple["Matrix", list[float]]:

        with open(file_path, 'r') as file:
            lines = file.readlines()
            n = int(lines[0].strip())

            matrix_data = []
            b_vector = []

            for i in range(1, n + 1):
                numbers = list(map(float, lines[i].strip().split()))

                if len(numbers) != n + 1:
                    raise ValueError(f"Ошибка в строке {i + 1}: ожидалось {n + 1} чисел, найдено {len(numbers)}.")

                matrix_data.append(numbers[:-1])

                b_vector.append(numbers[-1])

            return Matrix(matrix_data, n), b_vector

    @staticmethod
    def zeros(n: int) -> "Matrix":
        return Matrix([[0.0] * n for _ in range(n)], n)

    @staticmethod
    def eye(n: int) -> "Matrix":
        A = Matrix.zeros(n)
        for i in range(n):
            A.data[i][i] = 1.0
        return A

    @staticmethod
    def identity(n: int) -> "Matrix":
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)], n)

    def _check_same_size(self, other: "Matrix") -> None:
        if self.n != other.n:
            raise ValueError("Matrix sizes must match")

    def lu_decomposition(self) -> tuple["Matrix", "Matrix", list[int], int]:
        """
        Выполняет LU-разложение матрицы с выбором главного элемента (пивотингом).
        P*A = L*U
        Возвращает: (L, U, p, swaps)
        L - нижняя треугольная матрица
        U - верхняя треугольная матрица
        p - вектор перестановок
        swaps - количество перестановок
        """
        n = self.n
        L = Matrix.eye(n)
        U = self.copy()
        p = list(range(n))
        swaps = 0

        for k in range(n):

            pivot_val = 0
            pivot_row = -1
            for i in range(k, n):
                if abs(U.data[i][k]) > pivot_val:
                    pivot_val = abs(U.data[i][k])
                    pivot_row = i

            if pivot_val < 1e-9:
                raise ValueError("Матрица является сингулярной (вырожденной).")

            if pivot_row != k:
                U.data[k], U.data[pivot_row] = U.data[pivot_row], U.data[k]
                p[k], p[pivot_row] = p[pivot_row], p[k]

                L.data[k], L.data[pivot_row] = L.data[pivot_row], L.data[k]
                swaps += 1

            for i in range(k + 1, n):
                multiplier = U.data[i][k] / U.data[k][k]
                L.data[i][k] = multiplier
                for j in range(k, n):
                    U.data[i][j] -= multiplier * U.data[k][j]

        return L, U, p, swaps

    @staticmethod
    def _forward_substitution(L: "Matrix", b: list[float]) -> list[float]:
        n = len(L)
        y = [0.0] * n
        for i in range(n):
            sum_ly = sum(L.data[i][j] * y[j] for j in range(i))
            y[i] = b[i] - sum_ly
        return y

    @staticmethod
    def _backward_substitution(U: "Matrix", y: list[float]) -> list[float]:
        n = len(U)
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            sum_ux = sum(U.data[i][j] * x[j] for j in range(i + 1, n))
            x[i] = (y[i] - sum_ux) / U.data[i][i]
        return x

    def solve(self, b: list[float]) -> list[float]:
        # b -> Pb
        # Ly = Pb
        # Ly = Pb
        if len(b) != self.n:
            raise ValueError("Размер вектора b не совпадает с размером матрицы.")

        L, U, p, _ = self.lu_decomposition()

        b_permuted = [b[i] for i in p]

        y = self._forward_substitution(L, b_permuted)
        x = self._backward_substitution(U, y)

        return x

    def determinant(self) -> float:

        # det(A) = (-1)^swaps * det(U)

        try:
            _, U, _, swaps = self.lu_decomposition()
            det = math.pow(-1, swaps)
            for i in range(self.n):
                det *= U.data[i][i]
            return det
        except ValueError:
            return 0.0

    def inverse(self) -> "Matrix":

        n = self.n
        L, U, p, _ = self.lu_decomposition()

        inv_matrix = Matrix.zeros(n)
        ident = Matrix.identity(n)

        for j in range(n):
            e_j = [row[j] for row in ident.data]

            e_j_permuted = [e_j[i] for i in p]

            y = Matrix._forward_substitution(L, e_j_permuted)
            x = Matrix._backward_substitution(U, y)

            for i in range(n):
                inv_matrix.data[i][j] = x[i]

        return inv_matrix
