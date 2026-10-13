import numpy as np
import matplotlib.pyplot as plt

# AGUA
t = np.arange(28, dtype=float)
h = np.array([12.575, 11.951, 11.264, 10.665, 10.022, 9.451, 8.907, 8.310, 7.781, 7.283, 6.785, 6.287, 5.857, 5.374, 4.954, 4.560, 4.181, 3.434, 3.107, 2.760, 2.438, 2.168, 1.816, 1.660, 1.421, 1.131, 0.949, 0.737])
er = np.array([1.59, 1.63, 1.68, 1.73, 1.79, 1.85, 1.92, 2.00, 2.08, 2.17, 2.27, 2.39, 2.50, 2.66, 2.81, 2.99, 3.19, 3.71, 4.01, 4.42, 4.90, 5.41, 6.30, 6.82, 7.83, 9.64, 11.33, 14.36]) / 100

# <<< PLACEHOLDER: falta la tabla del ACEITE, completar t2, h2 y er2 >>>
t2 = np.array([])
h2 = np.array([])
er2 = np.array([]) / 100

liquidos = [('AGUA', t, h, er), ('ACEITE', t2, h2, er2)]

for nom, tt, hh, eer in liquidos:

    if len(tt) == 0:
        print('')
        print('!!!!!!!!  PLACEHOLDER: falta la tabla del ACEITE, este liquido se saltea  !!!!!!!!')
        print('!!!!!!!!  completar los arrays t2, h2 y er2 mas arriba del archivo            !!!!!!!!')
        print('')
        continue

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

    ecm2 = np.sqrt(np.mean((y - p2)**2))
    ecm3 = np.sqrt(np.mean((y - p3)**2))
    ecmexp = np.sqrt(np.mean((y - pexp)**2))

    plt.figure()
    plt.plot(tt, y, 'rx', markersize=5, label='mediciones')
    plt.plot(tt, p2, 'b-', label='cuadratico')
    plt.plot(tt, p3, color='violet', linestyle=':', label='cubico')
    plt.plot(tt, pexp, 'g--', label='exponencial')
    plt.xlabel('t [s]')
    plt.ylabel('h(t)/h0')
    plt.title(nom)
    plt.legend()
    plt.grid(True)
    plt.show()

    print('==== %s ====' % nom)
    print('puntos en el ajuste:', sel.sum(), 'de', len(tt))
    print('cuadratico :', a, b, c, ecm2)
    print('cubico     :', aC, bC, cC, dC, ecm3)
    print('exponencial:', aexp, bexp, ecmexp)
    
    #pending punto 5, hacerlo aca?
