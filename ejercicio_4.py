import os
import numpy as np
import matplotlib.pyplot as plt

# Parámetros para Punto 5 (Torricelli teórico)
d1 = 5.987
d2 = 0.478
g = 9.8

dir_graficos = os.path.join(os.path.dirname(__file__), 'graficos')
os.makedirs(dir_graficos, exist_ok=True)

# AGUA
t = np.arange(28, dtype=float)
h = np.array([12.6, 12.0, 11.3, 10.7, 10.0, 9.5, 8.9, 8.3, 7.8, 7.3,
              6.8, 6.3, 5.9, 5.4, 5.0, 4.6, 4.2, 3.4, 3.1, 2.8,
              2.4, 2.2, 1.8, 1.7, 1.4, 1.1, 0.9, 0.7])
er = np.array([1.59, 1.63, 1.68, 1.73, 1.79, 1.85, 1.92, 2.00, 2.08, 2.16,
               2.26, 2.38, 2.49, 2.65, 2.79, 2.97, 3.17, 3.73, 4.02, 4.37,
               4.96, 5.34, 6.35, 6.68, 7.94, 9.88, 11.90, 15.08]) / 100

# ACEITE
t2 = np.arange(26, dtype=float) * 1.2
h2 = np.array([13.0, 12.2, 11.5, 10.8, 10.1,  9.4,  8.7,  8.1,  7.6,  7.0,
                6.5,  6.0,  5.5,  5.0,  4.5,  4.1,  3.7,  3.3,  3.0,  2.7,
                2.4,  2.1,  1.8,  1.6,  1.4,  1.2])
er2 = np.array([1.54, 1.59, 1.64, 1.70, 1.76, 1.83, 1.92, 2.00, 2.09, 2.20,
                2.31, 2.44, 2.59, 2.77, 2.99, 3.21, 3.47, 3.80, 4.10, 4.47,
                4.94, 5.53, 6.32, 7.02, 7.91, 9.10]) / 100

liquidos = [('AGUA', t, h, er), ('ACEITE', t2, h2, er2)]

for nom, tt, hh, eer in liquidos:

    h0 = hh[0]
    y = hh / h0

    sel = eer < 0.10
    ta = tt[sel]
    ya = y[sel]

    uno = np.ones(len(ta))

    A2 = np.array([uno, ta, ta**2]).T
    a, b, c = np.linalg.solve(A2.T @ A2, A2.T @ ya)

    A3 = np.array([uno, ta, ta**2, ta**3]).T
    aC, bC, cC, dC = np.linalg.solve(A3.T @ A3, A3.T @ ya)

    A1 = np.array([uno, ta]).T
    aexp, mexp = np.linalg.solve(A1.T @ A1, A1.T @ np.log(ya))
    bexp = -mexp

    p2 = a + b*tt + c*tt**2
    p3 = aC + bC*tt + cC*tt**2 + dC*tt**3
    pexp = np.exp(aexp - bexp*tt)
    # Punto 5: estimación teórica ec. (1) y (2)
    tf = np.sqrt(2.0 * (h0 / 100.0) / g * ((d1 / d2)**4 - 1.0))
    p_teo = np.maximum(0.0, (1.0 - tt / tf)**2)

    ecm2 = np.sqrt(np.mean((y - p2)**2))
    ecm3 = np.sqrt(np.mean((y - p3)**2))
    ecmexp = np.sqrt(np.mean((y - pexp)**2))
    ecm_teo = np.sqrt(np.mean((y - p_teo)**2))

    plt.figure()
    plt.plot(tt, y, 'rx', markersize=5, label='mediciones')
    plt.plot(tt, p2, 'b-', label='cuadratico')
    plt.plot(tt, p3, color='violet', linestyle=':', label='cubico')
    plt.plot(tt, pexp, 'g--', label='exponencial')
    plt.plot(tt, p_teo, 'ko', markersize=4, label='teorico (ec. 1-2)')
    plt.xlabel('t [s]')
    plt.ylabel('h(t)/h0')
    plt.title(nom)
    plt.legend()
    plt.grid(True)

    ruta_png = os.path.join(dir_graficos, f'ajuste_{nom.lower()}.png')
    plt.savefig(ruta_png, dpi=300, bbox_inches='tight')
    plt.show()

    print('==== %s ====' % nom)
    print('puntos en el ajuste:', sel.sum(), 'de', len(tt))
    print('cuadratico :', a, b, c, ecm2)
    print('cubico     :', aC, bC, cC, dC, ecm3)
    print('exponencial:', aexp, bexp, ecmexp)
    print('teorico    : tf =', tf, 'ecm =', ecm_teo)
    print('grafico guardado en:', ruta_png)
    print()
