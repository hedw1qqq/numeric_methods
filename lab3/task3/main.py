# МНК (1-я и 2-я степени) для табличной функции
# Требуется ваш класс Matrix в модуле utility.matrix
# файл должен лежать как utility/matrix.py

from utility.Matrix import Matrix
import matplotlib.pyplot as plt

# Табличные точки (из задания)
x = [-1.0, 0.0, 1.0, 2.0, 3.0, 4.0]
y = [0.86603, 1.0, 0.86603, 0.50, 0.0, -0.50]

n = len(x)

# Предварительные суммы
Sx   = sum(x)
Sx2  = sum(t*t for t in x)
Sx3  = sum(t**3 for t in x)
Sx4  = sum(t**4 for t in x)
Sy   = sum(y)
Sxy  = sum(x[i]*y[i] for i in range(n))
Sx2y = sum((x[i]**2)*y[i] for i in range(n))

# 1) Линейный МНК: P1(x) = a0 + a1 x
A1 = Matrix([[n,  Sx],
             [Sx, Sx2]], 2)
b1 = [Sy, Sxy]
a0_lin, a1_lin = A1.solve(b1)

def P1(t): return a0_lin + a1_lin*t
SSE1 = sum((y[i] - P1(x[i]))**2 for i in range(n))

# 2) Квадратичный МНК: P2(x) = a0 + a1 x + a2 x^2
A2 = Matrix([[n,   Sx,  Sx2],
             [Sx,  Sx2, Sx3],
             [Sx2, Sx3, Sx4]], 3)
b2 = [Sy, Sxy, Sx2y]
a0_quad, a1_quad, a2_quad = A2.solve(b2)

def P2(t): return a0_quad + a1_quad*t + a2_quad*t*t
SSE2 = sum((y[i] - P2(x[i]))**2 for i in range(n))

print("Линейный:  a0=%.12f, a1=%.12f, SSE=%.12f" % (a0_lin, a1_lin, SSE1))
print("Квадратичный: a0=%.12f, a1=%.12f, a2=%.12f, SSE=%.12f"
      % (a0_quad, a1_quad, a2_quad, SSE2))

# График
xs = [min(x) + (max(x)-min(x))*k/300.0 for k in range(301)]
y1 = [P1(t) for t in xs]
y2 = [P2(t) for t in xs]

plt.figure(figsize=(7,4))
plt.scatter(x, y, color="tab:red", label="табличные точки", zorder=3)
plt.plot(xs, y1, color="tab:blue", label="МНК: степень 1")
plt.plot(xs, y2, color="tab:green", label="МНК: степень 2")
plt.title("Аппроксимация по МНК (Matrix, LU)")
plt.xlabel("x"); plt.ylabel("y")
plt.grid(True, alpha=0.3); plt.legend()
plt.tight_layout(); plt.show()
