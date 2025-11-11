x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 0.86603, 0.5, 0.0, -0.5]
X_star = 1.5


def natural_cubic_spline_seconds(x, y):
    """
     Вычисляет вторые производные естественного кубического сплайна в узлах.

     Метод решения:
     Вторые производные m_i = S''(x_i) находятся из трёхдиагональной системы
     линейных уравнений (формулы 3.13 методички):

     h_{i-1}·m_{i-1} + 2(h_{i-1} + h_i)·m_i + h_i·m_{i+1} =
         6·[(f_{i+1} - f_i)/h_i - (f_i - f_{i-1})/h_{i-1}]

     для i = 1, ..., n-1, где h_i = x_{i+1} - x_i.
     Система решается методом прогонки (Thomas algorithm):
     """
    n = len(x) - 1
    h = [x[i + 1] - x[i] for i in range(n)]
    if n < 2:
        return [0.0] * (n + 1)
    # Коэффициенты трёхдиагональной матрицы
    a = [0.0] * (n - 1)  # поддиагональ
    b = [0.0] * (n - 1)  # главная
    c = [0.0] * (n - 1)  # наддиагональ
    d = [0.0] * (n - 1)  # правая часть

    for i in range(1, n):
        j = i - 1
        a[j] = h[i - 1]
        b[j] = 2 * (h[i - 1] + h[i])
        c[j] = h[i]
        d[j] = 6 * ((y[i + 1] - y[i]) / h[i] - (y[i] - y[i - 1]) / h[i - 1])

    for i in range(1, n - 1):
        w = a[i] / b[i - 1]
        b[i] -= w * c[i - 1]
        d[i] -= w * d[i - 1]

    m_inner = [0.0] * (n - 1)
    m_inner[-1] = d[-1] / b[-1]
    for i in range(n - 3, -1, -1):
        m_inner[i] = (d[i] - c[i] * m_inner[i + 1]) / b[i]

    m = [0.0] * (n + 1)
    for i in range(1, n):
        m[i] = m_inner[i - 1]
    return m


def spline_eval(x, y, m, t):
    """
       Вычисляет значение кубического сплайна в точке t.
       На отрезке [x_i, x_{i+1}] сплайн задаётся формулой (раздел 3.1 методички):

       S_i(t) = m_i·(x_{i+1} - t)³/(6·h_i) + m_{i+1}·(t - x_i)³/(6·h_i)
              + [y_i - m_i·h_i²/6]·(x_{i+1} - t)/h_i
              + [y_{i+1} - m_{i+1}·h_i²/6]·(t - x_i)/h_i

       где h_i = x_{i+1} - x_i.
       """
    if t <= x[0]:
        i = 0
    elif t >= x[-1]:
        i = len(x) - 2
    else:
        i = 0
        while not (x[i] <= t <= x[i + 1]):
            i += 1
    h = x[i + 1] - x[i]
    xi, xi1 = x[i], x[i + 1]
    yi, yi1 = y[i], y[i + 1]
    mi, mi1 = m[i], m[i + 1]
    S = (mi * (xi1 - t) ** 3 + mi1 * (t - xi) ** 3) / (6 * h) \
        + (yi - mi * h * h / 6.0) * (xi1 - t) / h \
        + (yi1 - mi1 * h * h / 6.0) * (t - xi) / h
    return S


m = natural_cubic_spline_seconds(x, y)
S_xstar = spline_eval(x, y, m, X_star)
print("S(1.5) =", S_xstar)

import matplotlib.pyplot as plt

xs = []
ys = []
for i in range(len(x) - 1):
    t0, t1 = x[i], x[i + 1]
    steps = 50
    for k in range(steps + 1):
        t = t0 + (t1 - t0) * k / steps
        xs.append(t)
        ys.append(spline_eval(x, y, m, t))

plt.figure(figsize=(7, 4))
plt.plot(xs, ys, label="Кубический сплайн", color="tab:blue")
plt.scatter(x, y, label="Узлы", color="tab:red", zorder=3)
plt.scatter([X_star], [S_xstar], label="S(1.5)", color="tab:green", zorder=4)
plt.title("Естественный кубический сплайн")
plt.xlabel("x")
plt.ylabel("S(x)")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
