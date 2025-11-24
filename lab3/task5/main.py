import math


def f(x: float) -> float:
    return x / (3 * x + 4) ** 2


def integral_rect(a: float, b: float, h: float) -> float:
    n = int((b - a) / h)
    s = 0.0
    for i in range(n):
        x_mid = a + (i + 0.5) * h
        s += f(x_mid)
    return s * h


def integral_trap(a: float, b: float, h: float) -> float:
    n = int((b - a) / h)
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        x_i = a + i * h
        s += f(x_i)
    return s * h


def integral_simp(a: float, b: float, h: float) -> float:

    n = int((b - a) / h)
    if n % 2 != 0:
        raise ValueError("Для Симпсона нужно чётное число интервалов")
    s = f(a) + f(b)
    for i in range(1, n):
        x_i = a + i * h
        s += (4 if i % 2 == 1 else 2) * f(x_i)
    return s * h / 3.0


def runge_romberg(F_h1: float, F_h2: float, k: float, p: int) -> float:
    return F_h2 + (F_h2 - F_h1) / (k ** p - 1.0)


a, b = 0.0, 4.0
h1, h2 = 1.0, 0.5
k = h1 / h2  # = 2

# h1
R1 = integral_rect(a, b, h1)
T1 = integral_trap(a, b, h1)
S1 = integral_simp(a, b, h1)

# h2
R2 = integral_rect(a, b, h2)
T2 = integral_trap(a, b, h2)
S2 = integral_simp(a, b, h2)

F_rect_RR = runge_romberg(R1, R2, k, p=2)
F_trap_RR = runge_romberg(T1, T2, k, p=2)
F_simp_RR = runge_romberg(S1, S2, k, p=4)

exact = 0.07069937

print("=" * 70)
print("РЕЗУЛЬТАТЫ ЧИСЛЕННОГО ИНТЕГРИРОВАНИЯ")
print("=" * 70)
print(f"Точное значение интеграла: {exact:.8f}\n")

print(f"{'Метод':<20} {'F(h₁=1.0)':<15} {'F(h₂=0.5)':<15} {'Рунге-Ромберг':<15}")
print("-" * 70)
print(f"{'Прямоугольники':<20} {R1:<15.8f} {R2:<15.8f} {F_rect_RR:<15.8f}")
print(f"{'Трапеции':<20} {T1:<15.8f} {T2:<15.8f} {F_trap_RR:<15.8f}")
print(f"{'Симпсон':<20} {S1:<15.8f} {S2:<15.8f} {F_simp_RR:<15.8f}")
print("=" * 70)

print("\nПОГРЕШНОСТИ (Рунге-Ромберг):")
print(f"{'Метод':<20} {'Абсолютная':<15} {'Относительная, %':<15}")
print("-" * 50)
print(f"{'Прямоугольники':<20} {abs(F_rect_RR - exact):<15.2e} {abs(F_rect_RR - exact)/exact*100:<15.4f}")
print(f"{'Трапеции':<20} {abs(F_trap_RR - exact):<15.2e} {abs(F_trap_RR - exact)/exact*100:<15.4f}")
print(f"{'Симпсон':<20} {abs(F_simp_RR - exact):<15.2e} {abs(F_simp_RR - exact)/exact*100:<15.4f}")