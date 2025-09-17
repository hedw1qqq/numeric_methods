from utility.Matrix import Matrix
import numpy as np

if __name__ == '__main__':
    A, b = Matrix.read_system_from_file("inputy")
    answer = A.solve(b)
    A_np = np.array(A.data)
    b_np = np.array(b)
    x_np = np.linalg.solve(A_np, b_np)
    print(f"SOLVE\nmy test\n{answer}\nnp test\n{x_np}")
    print("-" * 40)
    det_A = A.determinant()
    print(f"det A\nmy test =  {det_A:.4f}\nnp test = {np.linalg.det(A_np):.4f}")
    print("-" * 40)

    A_inv = A.inverse()
    print("Обратная матрица A⁻¹:")
    print(A_inv)

    print("invert = ", )
    print(np.linalg.inv(A_np))
