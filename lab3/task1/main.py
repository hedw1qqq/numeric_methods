import math


def lagrange_eval(xs, ys, t):
    """
    Вычисляет значение интерполяционного полинома Лагранжа в точке t.

    Формула Лагранжа (3.5 из методички):
    L_n(t) = Σ(i=0 to n) y_i · ℓ_i(t)

    xs - список узлов интерполяции
    ys - список значений функции
    t  - точка, в которой вычисляем полином

    """
    n = len(xs)
    s = 0.0
    for i in range(n):
        li = 1.0
        xi = xs[i]
        for j in range(n):
            if j == i:
                continue
            li *= (t - xs[j]) / (xi - xs[j])
        s += ys[i] * li
    return s


def newton_divdiff(xs, ys):
    """
    Вычисляет коэффициенты полинома Ньютона через разделённые разности.

    Полином Ньютона (3.8 из методички):
    P_n(x) = [x_0] + [x_0,x_1]·(x-x_0) + [x_0,x_1,x_2]·(x-x_0)(x-x_1) + ...

    Разделённые разности:
    [x_i] = f(x_i)                                    (порядок 0)
    [x_i,x_j] = (f(x_j) - f(x_i))/(x_j - x_i)        (порядок 1)
    [x_i,...,x_k] = ([x_{i+1},...,x_k] - [x_i,...,x_{k-1}])/(x_k - x_i)

    Таблица разделённых разностей (табл. 3.1 методички):
    x_0  y_0  [x_0,x_1]  [x_0,x_1,x_2]  [x_0,x_1,x_2,x_3]
    x_1  y_1  [x_1,x_2]  [x_1,x_2,x_3]
    x_2  y_2  [x_2,x_3]
    x_3  y_3

    Параметры:
    xs - список узлов [x_0, x_1, ..., x_n]
    ys - список значений [f(x_0), f(x_1), ..., f(x_n)]

    Возвращает: [a_0, a_1, ..., a_n], где a_k = [x_0,...,x_k]
    """
    a = ys[:]
    n = len(xs)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            a[i] = (a[i] - a[i - 1]) / (xs[i] - xs[i - j])
    return a


def newton_eval(xs, a, t):
    """
    Вычисляет значение полинома Ньютона в точке t по схеме Горнера.

    Полином Ньютона:
    P_n(t) = a_0 + a_1·(t-x_0) + a_2·(t-x_0)(t-x_1) + ... + a_n·Π(t-x_i)

    xs - узлы
    a  - коэффициенты  из newton_divdiff
    t  - точка вычисления

    """
    s = a[-1]
    for k in range(len(a) - 2, -1, -1):
        s = s * (t - xs[k]) + a[k]
    return s


def remainder_bound(xs, t, M):
    """
    Вычисляет теоретическую верхнюю границу погрешности интерполяции.

    Формула остаточного члена (3.9 из методички):
    |f(t) - P_n(t)| ≤ M/(n+1)! · |ω_{n+1}(t)|

    где:
    - M = max|f^{(n+1)}(ξ)| на отрезке [x_0, x_n]
    - ω_{n+1}(t) = Π(i=0 to n) (t - x_i) - узловое произведение
    - n+1 = количество узлов (степень полинома = n)

    xs - узлы интерполяции
    t  - точка, в которой оцениваем погрешность
    M  - максимум модуля производной порядка (n+1)
    """
    prod = 1.0
    for x in xs:
        prod *= abs(t - x)
    return M * prod / math.factorial(len(xs))


pi = math.pi
f = math.cos
x_star = pi / 4

# variant (a)
Xa = [0.0, pi / 6, 2 * pi / 6, 3 * pi / 6]
Ya = [f(x) for x in Xa]

# variant (b)
Xb = [0.0, pi / 6, 5 * pi / 12, pi / 2]
Yb = [f(x) for x in Xb]


def report(xs, ys, name):
    print(f"--- {name} ---")

    L = lagrange_eval(xs, ys, x_star)
    a = newton_divdiff(xs, ys)
    N = newton_eval(xs, a, x_star)

    true = f(x_star)
    err_L = abs(true - L)
    err_N = abs(true - N)

    bound = remainder_bound(xs, x_star, M=1.0)
    print(f"Лагранж в π/4: {L:.12f}, |err|={err_L:.12e}")
    print(f"Ньютон в π/4: {N:.12f}, |err|={err_N:.12e}")
    print(f"теоретическая верхняя граница погрешности интерполяции ≤ {bound:.12e}")


report(Xa, Ya, "Variant (a)")
report(Xb, Yb, "Variant (b)")
