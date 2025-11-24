X = [-1.0, 0.0, 1.0, 2.0, 3.0]
Y = [-0.5, 0.0, 0.5, 0.86603, 1.0]
X_STAR = 1.0
eps = 1e-6


def calculate_first_derivative(X, Y, X_STAR):
    idx = None
    for i, xi in enumerate(X):
        if abs(xi - X_STAR) < eps:
            idx = i
            break

    if idx is None:
        raise ValueError

    if idx == 0:
        return (Y[1] - Y[0]) / (X[1] - X[0])
    if idx == len(X) - 1:
        return (Y[idx] - Y[idx - 1]) / (X[idx] - X[idx - 1])
    idx -= 1
    first_fraction = (Y[idx + 1] - Y[idx]) / (X[idx + 1] - X[idx])
    s1 = (Y[idx + 2] - Y[idx + 1]) / (X[idx + 2] - X[idx + 1])
    s2 = (Y[idx + 1] - Y[idx]) / (X[idx + 1] - X[idx])
    second_fraction = (s1 - s2) / (X[idx + 2] - X[idx])
    return first_fraction + second_fraction * (2 * X_STAR - X[idx] - X[idx + 1])


def calculate_second_derivative(X, Y, X_STAR):
    idx = None
    for i, xi in enumerate(X):
        if abs(xi - X_STAR) < eps:
            idx = i
            break

    if idx is None or idx == len(X) - 1 or idx == 0:
        raise ValueError
    idx -= 1
    s1 = (Y[idx + 2] - Y[idx + 1]) / (X[idx + 2] - X[idx + 1])
    s2 = (Y[idx + 1] - Y[idx]) / (X[idx + 1] - X[idx])
    fraction = (s1 - s2) / (X[idx + 2] - X[idx])
    return 2 * fraction


print(f"Значение первой производной в {X_STAR} - {calculate_first_derivative(X, Y, X_STAR)}")
print(f"Значение второй производной в {X_STAR} - {calculate_second_derivative(X, Y, X_STAR)}")
