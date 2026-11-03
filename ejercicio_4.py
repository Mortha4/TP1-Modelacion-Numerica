import numpy as np
import matplotlib.pyplot as plt

# Parametros para Punto 5 (Torricelli teorico)
d1 = 5.987
d2 = 0.478
g = 9.8
dh = 0.2

# AGUA (30 mediciones, cada 0.6 s)
t = np.array([0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4,
              6.0, 6.6, 7.2, 7.8, 8.4, 9.0, 9.6, 10.2, 10.8, 11.4,
              12.0, 12.6, 13.2, 13.8, 14.4, 15.0, 15.6, 16.2, 16.8, 17.4])
h = np.array([12.2, 11.8, 11.5, 11.1, 10.7, 10.3, 10.0, 9.6, 9.3, 9.0,
              8.6, 8.3, 8.0, 7.6, 7.3, 7.0, 6.7, 6.5, 6.2, 5.9,
              5.6, 5.4, 5.1, 4.9, 4.6, 4.4, 4.1, 3.9, 3.7, 3.5])

# ACEITE (26 mediciones, cada 0.7 s)
t2 = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3,
               7.0, 7.7, 8.4, 9.1, 9.8, 10.5, 11.2, 11.9, 12.6, 13.3,
               14.0, 14.7, 15.4, 16.1, 16.8, 17.5])
h2 = np.array([13.0, 12.7, 12.2, 11.8, 11.4, 10.9, 10.5, 10.2, 9.8, 9.4,
               9.0, 8.6, 8.3, 8.0, 7.6, 7.3, 7.0, 6.7, 6.4, 6.1,
               5.5, 5.2, 5.0, 4.2, 4.0, 3.7])

# ==========================================
# AGUA
# ==========================================
h0 = h[0]
y = h / h0
dy = (dh / h0) + (h * dh) / (h0 ** 2)
er = dy / y

# Filtrar solo mediciones con error relativo menor al 10% (0.10)
sel = er < 0.10
ta = t[sel]
ya = y[sel]

# Funciones base: phi0 = 1, phi1 = t, phi2 = t^2, phi3 = t^3
phi0 = np.ones(ta.size)
phi1 = ta
phi2 = ta ** 2
phi3 = ta ** 3

# a) Ajuste cuadratico: y = a + b*t + c*t^2
M2 = np.array([
    [np.sum(phi0 * phi0), np.sum(phi0 * phi1), np.sum(phi0 * phi2)],
    [np.sum(phi1 * phi0), np.sum(phi1 * phi1), np.sum(phi1 * phi2)],
    [np.sum(phi2 * phi0), np.sum(phi2 * phi1), np.sum(phi2 * phi2)]
])
b2 = np.array([np.sum(ya * phi0), np.sum(ya * phi1), np.sum(ya * phi2)])
a, b, c = np.linalg.solve(M2, b2)

# b) Ajuste cubico: y = aC + bC*t + cC*t^2 + dC*t^3
M3 = np.array([
    [np.sum(phi0 * phi0), np.sum(phi0 * phi1), np.sum(phi0 * phi2), np.sum(phi0 * phi3)],
    [np.sum(phi1 * phi0), np.sum(phi1 * phi1), np.sum(phi1 * phi2), np.sum(phi1 * phi3)],
    [np.sum(phi2 * phi0), np.sum(phi2 * phi1), np.sum(phi2 * phi2), np.sum(phi2 * phi3)],
    [np.sum(phi3 * phi0), np.sum(phi3 * phi1), np.sum(phi3 * phi2), np.sum(phi3 * phi3)]
])
b3 = np.array([np.sum(ya * phi0), np.sum(ya * phi1), np.sum(ya * phi2), np.sum(ya * phi3)])
aC, bC, cC, dC = np.linalg.solve(M3, b3)

# c) Ajuste exponencial linealizado: ln(y) = aexp - bexp*t
Y_lin = np.log(ya)
M1 = np.array([
    [np.sum(phi0 * phi0), np.sum(phi0 * phi1)],
    [np.sum(phi1 * phi0), np.sum(phi1 * phi1)]
])
b1 = np.array([np.sum(Y_lin * phi0), np.sum(Y_lin * phi1)])
aexp, mexp = np.linalg.solve(M1, b1)
bexp = -mexp

# Evaluacion en TODOS los tiempos medidos
p2 = a + b * t + c * (t ** 2)
p3 = aC + bC * t + cC * (t ** 2) + dC * (t ** 3)
pexp = np.exp(aexp - bexp * t)

# Punto 5: estimacion teorica ec. (1) y (2)
tf = np.sqrt(2.0 * (h0 / 100.0) / g * ((d1 / d2) ** 4 - 1.0))
p_teo = (1.0 - t / tf) ** 2

# Error cuadratico medio (ECM) tomando en cuenta TODAS las mediciones
ecm2 = np.sqrt(np.sum((y - p2) ** 2) / y.size)
ecm3 = np.sqrt(np.sum((y - p3) ** 2) / y.size)
ecmexp = np.sqrt(np.sum((y - pexp) ** 2) / y.size)
ecm_teo = np.sqrt(np.sum((y - p_teo) ** 2) / y.size)

plt.figure()
plt.plot(t, y, 'rx', markersize=5, label='mediciones')
plt.plot(t, p2, 'b-', label='cuadratico')
plt.plot(t, p3, color='violet', linestyle=':', label='cubico')
plt.plot(t, pexp, 'g--', label='exponencial')
plt.plot(t, p_teo, 'ko', markersize=4, label='teorico (ec. 1-2)')
plt.xlabel('t [s]')
plt.ylabel('h(t)/h0')
plt.title('AGUA')
plt.legend()
plt.grid(True)

ruta_png_agua = 'graficos/ajuste_agua.png'
plt.savefig(ruta_png_agua, dpi=300, bbox_inches='tight')
plt.show()

print('==== AGUA ====')
print('puntos en el ajuste:', ta.size, 'de', t.size)
print('cuadratico :', a, b, c, 'ECM =', ecm2)
print('cubico     :', aC, bC, cC, dC, 'ECM =', ecm3)
print('exponencial:', aexp, bexp, 'ECM =', ecmexp)
print('teorico    : tf =', tf, 'ECM =', ecm_teo)
print('grafico guardado en:', ruta_png_agua)
print()


# ==========================================
# ACEITE
# ==========================================
h0_2 = h2[0]
y2 = h2 / h0_2
dy2 = (dh / h0_2) + (h2 * dh) / (h0_2 ** 2)
er2 = dy2 / y2

# Filtrar solo mediciones con error relativo menor al 10% (0.10)
sel2 = er2 < 0.10
ta2 = t2[sel2]
ya2 = y2[sel2]

# Funciones base: phi0 = 1, phi1 = t, phi2 = t^2, phi3 = t^3
phi0_2 = np.ones(ta2.size)
phi1_2 = ta2
phi2_2 = ta2 ** 2
phi3_2 = ta2 ** 3

# a) Ajuste cuadratico: y = a + b*t + c*t^2
M2_2 = np.array([
    [np.sum(phi0_2 * phi0_2), np.sum(phi0_2 * phi1_2), np.sum(phi0_2 * phi2_2)],
    [np.sum(phi1_2 * phi0_2), np.sum(phi1_2 * phi1_2), np.sum(phi1_2 * phi2_2)],
    [np.sum(phi2_2 * phi0_2), np.sum(phi2_2 * phi1_2), np.sum(phi2_2 * phi2_2)]
])
b2_2 = np.array([np.sum(ya2 * phi0_2), np.sum(ya2 * phi1_2), np.sum(ya2 * phi2_2)])
a_2, b_2, c_2 = np.linalg.solve(M2_2, b2_2)

# b) Ajuste cubico: y = aC + bC*t + cC*t^2 + dC*t^3
M3_2 = np.array([
    [np.sum(phi0_2 * phi0_2), np.sum(phi0_2 * phi1_2), np.sum(phi0_2 * phi2_2), np.sum(phi0_2 * phi3_2)],
    [np.sum(phi1_2 * phi0_2), np.sum(phi1_2 * phi1_2), np.sum(phi1_2 * phi2_2), np.sum(phi1_2 * phi3_2)],
    [np.sum(phi2_2 * phi0_2), np.sum(phi2_2 * phi1_2), np.sum(phi2_2 * phi2_2), np.sum(phi2_2 * phi3_2)],
    [np.sum(phi3_2 * phi0_2), np.sum(phi3_2 * phi1_2), np.sum(phi3_2 * phi2_2), np.sum(phi3_2 * phi3_2)]
])
b3_2 = np.array([np.sum(ya2 * phi0_2), np.sum(ya2 * phi1_2), np.sum(ya2 * phi2_2), np.sum(ya2 * phi3_2)])
aC_2, bC_2, cC_2, dC_2 = np.linalg.solve(M3_2, b3_2)

# c) Ajuste exponencial linealizado: ln(y) = aexp - bexp*t
Y_lin_2 = np.log(ya2)
M1_2 = np.array([
    [np.sum(phi0_2 * phi0_2), np.sum(phi0_2 * phi1_2)],
    [np.sum(phi1_2 * phi0_2), np.sum(phi1_2 * phi1_2)]
])
b1_2 = np.array([np.sum(Y_lin_2 * phi0_2), np.sum(Y_lin_2 * phi1_2)])
aexp_2, mexp_2 = np.linalg.solve(M1_2, b1_2)
bexp_2 = -mexp_2

# Evaluacion en TODOS los tiempos medidos
p2_2 = a_2 + b_2 * t2 + c_2 * (t2 ** 2)
p3_2 = aC_2 + bC_2 * t2 + cC_2 * (t2 ** 2) + dC_2 * (t2 ** 3)
pexp_2 = np.exp(aexp_2 - bexp_2 * t2)

# Punto 5: estimacion teorica ec. (1) y (2)
tf_2 = np.sqrt(2.0 * (h0_2 / 100.0) / g * ((d1 / d2) ** 4 - 1.0))
p_teo_2 = (1.0 - t2 / tf_2) ** 2

# Error cuadratico medio (ECM) tomando en cuenta TODAS las mediciones
ecm2_2 = np.sqrt(np.sum((y2 - p2_2) ** 2) / y2.size)
ecm3_2 = np.sqrt(np.sum((y2 - p3_2) ** 2) / y2.size)
ecmexp_2 = np.sqrt(np.sum((y2 - pexp_2) ** 2) / y2.size)
ecm_teo_2 = np.sqrt(np.sum((y2 - p_teo_2) ** 2) / y2.size)

plt.figure()
plt.plot(t2, y2, 'rx', markersize=5, label='mediciones')
plt.plot(t2, p2_2, 'b-', label='cuadratico')
plt.plot(t2, p3_2, color='violet', linestyle=':', label='cubico')
plt.plot(t2, pexp_2, 'g--', label='exponencial')
plt.plot(t2, p_teo_2, 'ko', markersize=4, label='teorico (ec. 1-2)')
plt.xlabel('t [s]')
plt.ylabel('h(t)/h0')
plt.title('ACEITE')
plt.legend()
plt.grid(True)

ruta_png_aceite = 'graficos/ajuste_aceite.png'
plt.savefig(ruta_png_aceite, dpi=300, bbox_inches='tight')
plt.show()

print('==== ACEITE ====')
print('puntos en el ajuste:', ta2.size, 'de', t2.size)
print('cuadratico :', a_2, b_2, c_2, 'ECM =', ecm2_2)
print('cubico     :', aC_2, bC_2, cC_2, dC_2, 'ECM =', ecm3_2)
print('exponencial:', aexp_2, bexp_2, 'ECM =', ecmexp_2)
print('teorico    : tf =', tf_2, 'ECM =', ecm_teo_2)
print('grafico guardado en:', ruta_png_aceite)
print()

