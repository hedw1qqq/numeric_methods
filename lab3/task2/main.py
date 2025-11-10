
x = [0.0, 1.0, 2.0, 3.0, 4.0]
y = [1.0, 0.86603, 0.5, 0.0, -0.5]
X_star = 1.5

# 1) собираем трёхдиагональную СЛАУ для m_i = S''(x_i) при m_0 = m_n = 0
def natural_cubic_spline_seconds(x, y):
    n = len(x) - 1
    h = [x[i+1]-x[i] for i in range(n)]
    if n < 2:
        return [0.0]*(n+1)
    # Коэффициенты трёхдиагональной матрицы (Thomas)
    a = [0.0]*(n-1)   # поддиагональ
    b = [0.0]*(n-1)   # главная
    c = [0.0]*(n-1)   # наддиагональ
    d = [0.0]*(n-1)   # правая часть
    for i in range(1, n):
        j = i - 1
        a[j] = h[i-1]
        b[j] = 2*(h[i-1] + h[i])
        c[j] = h[i]
        d[j] = 6*((y[i+1]-y[i])/h[i] - (y[i]-y[i-1])/h[i-1])

    # Прямой ход
    for i in range(1, n-1):
        w = a[i] / b[i-1]
        b[i] -= w * c[i-1]
        d[i] -= w * d[i-1]
    # Обратный ход
    m_inner = [0.0]*(n-1)
    m_inner[-1] = d[-1] / b[-1]
    for i in range(n-3, -1, -1):
        m_inner[i] = (d[i] - c[i]*m_inner[i+1]) / b[i]

    # Собираем m с краями 0
    m = [0.0]*(n+1)
    for i in range(1, n):
        m[i] = m_inner[i-1]
    return m

def spline_eval(x, y, m, t):
    # Находим отрезок [x_i, x_{i+1}]
    if t <= x[0]:
        i = 0
    elif t >= x[-1]:
        i = len(x)-2
    else:
        i = 0
        while not (x[i] <= t <= x[i+1]):
            i += 1
    h = x[i+1] - x[i]
    xi, xi1 = x[i], x[i+1]
    yi, yi1 = y[i], y[i+1]
    mi, mi1 = m[i], m[i+1]
    # Формула локального куба из методички
    S = (mi*(xi1 - t)**3 + mi1*(t - xi)**3)/(6*h) \
        + (yi - mi*h*h/6.0)*(xi1 - t)/h \
        + (yi1 - mi1*h*h/6.0)*(t - xi)/h
    return S

m = natural_cubic_spline_seconds(x, y)
S_xstar = spline_eval(x, y, m, X_star)
print("S(1.5) =", S_xstar)

# Построение графика (Matplotlib, без numpy)
import matplotlib.pyplot as plt

# Точки для графика сплайна
xs = []
ys = []
# равномерно по каждому отрезку
for i in range(len(x)-1):
    t0, t1 = x[i], x[i+1]
    steps = 50
    for k in range(steps+1):
        t = t0 + (t1 - t0)*k/steps
        xs.append(t)
        ys.append(spline_eval(x, y, m, t))

plt.figure(figsize=(7,4))
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
