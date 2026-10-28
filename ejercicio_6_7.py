import numpy as np

d1, d2, g = 5.987, 0.478, 9.8

def calcular_tf(h0_cm):
    return np.sqrt(2.0 * (h0_cm / 100.0) / g * ((d1 / d2)**4 - 1.0))

# AGUA
t_agua = np.arange(28, dtype=float)
h_agua = np.array([12.6, 12.0, 11.3, 10.7, 10.0, 9.5, 8.9, 8.3, 7.8, 7.3,
                   6.8, 6.3, 5.9, 5.4, 5.0, 4.6, 4.2, 3.4, 3.1, 2.8,
                   2.4, 2.2, 1.8, 1.7, 1.4, 1.1, 0.9, 0.7])
er_agua = np.array([1.59, 1.63, 1.68, 1.73, 1.79, 1.85, 1.92, 2.00, 2.08, 2.16,
                    2.26, 2.38, 2.49, 2.65, 2.79, 2.97, 3.17, 3.73, 4.02, 4.37,
                    4.96, 5.34, 6.35, 6.68, 7.94, 9.88, 11.90, 15.08]) / 100

# ACEITE
t_aceite = np.arange(26, dtype=float) * 1.2
h_aceite = np.array([13.0, 12.2, 11.5, 10.8, 10.1,  9.4,  8.7,  8.1,  7.6,  7.0,
                     6.5,  6.0,  5.5,  5.0,  4.5,  4.1,  3.7,  3.3,  3.0,  2.7,
                     2.4,  2.1,  1.8,  1.6,  1.4,  1.2])
er_aceite = np.array([1.54, 1.59, 1.64, 1.70, 1.76, 1.83, 1.92, 2.00, 2.09, 2.20,
                      2.31, 2.44, 2.59, 2.77, 2.99, 3.21, 3.47, 3.80, 4.10, 4.47,
                      4.94, 5.53, 6.32, 7.02, 7.91, 9.10]) / 100

liquidos = [('AGUA', t_agua, h_agua, er_agua), ('ACEITE', t_aceite, h_aceite, er_aceite)]

obj = [0.50, 0.10]

for nom, t, h, er in liquidos:
    print('==== %s ====' % nom)
    y = h / h[0]
    tf = calcular_tf(h[0])

    sel = er < 0.10
    ta = t[sel]
    ya = y[sel]
    uno = np.ones(len(ta))
    A3 = np.array([uno, ta, ta**2, ta**3]).T
    aC, bC, cC, dC = np.linalg.solve(A3.T @ A3, A3.T @ ya)

    for o in obj:
        f = lambda x: aC + bC*x + cC*x**2 + dC*x**3 - o
        df = lambda x: bC + 2*cC*x + 3*dC*x**2

        x = 10.0
        tol = 0.5e-9
        n = 0
        x1 = x - f(x)/df(x)
        while abs(x1 - x) > tol:
            x = x1
            x1 = x - f(x)/df(x)
            n = n + 1

        tt = tf * (1 - np.sqrt(o))

        i = 0
        while y[i] > o:
            i = i + 1
        te = t[i-1] + (y[i-1] - o)/(y[i-1] - y[i])*(t[i] - t[i-1])

        print('objetivo y =', o)
        print('  ec. (1) y (2)  ', tt)
        print('  cubico        ', x1)
        print('  experimental  ', te)
        print('  diferencia    ', x1 - tt, '   iteraciones', n)
        print()