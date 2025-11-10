import math

def lagrange_eval(xs, ys, t):
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
    a = ys[:]
    n = len(xs)
    for j in range(1, n):
        for i in range(n-1, j-1, -1):
            a[i] = (a[i] - a[i-1]) / (xs[i] - xs[i-j])
    return a  # a[k] = [x0,...,xk]

def newton_eval(xs, a, t):
    # Horner-like evaluation in Newton basis
    s = a[-1]
    for k in range(len(a)-2, -1, -1):
        s = s * (t - xs[k]) + a[k]
    return s

def remainder_bound(xs, t, M):
    # |R_n(t)| <= M/(n+1)! * prod |t - x_i|
    prod = 1.0
    for x in xs:
        prod *= abs(t - x)
    return M * prod / math.factorial(len(xs))

# Problem settings
pi = math.pi
f = math.cos
x_star = pi / 4

# variant (a)
Xa = [0.0, pi/6, 2*pi/6, 3*pi/6]
Ya = [f(x) for x in Xa]

# variant (b)
Xb = [0.0, pi/6, 5*pi/12, pi/2]
Yb = [f(x) for x in Xb]

def report(xs, ys, name):
    print(f"--- {name} ---")
    # Lagrange
    L = lagrange_eval(xs, ys, x_star)
    # Newton
    a = newton_divdiff(xs, ys)
    N = newton_eval(xs, a, x_star)
    # true value and errors
    true = f(x_star)
    err_L = abs(true - L)
    err_N = abs(true - N)
    # remainder bound with M=max|f^{(n+1)}|=1 for cos, n+1=len(xs)
    bound = remainder_bound(xs, x_star, M=1.0)
    print(f"Lagrange at π/4: {L:.12f}, |err|={err_L:.12e}")
    print(f"Newton   at π/4: {N:.12f}, |err|={err_N:.12e}")
    print(f"Theoretical bound ≤ {bound:.12e}")

report(Xa, Ya, "Variant (a)")
report(Xb, Yb, "Variant (b)")
