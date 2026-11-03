import numpy as np


# AGUA (30 mediciones, cada 0.6 s)
t_agua = np.array([0.0, 0.6, 1.2, 1.8, 2.4, 3.0, 3.6, 4.2, 4.8, 5.4,
                   6.0, 6.6, 7.2, 7.8, 8.4, 9.0, 9.6, 10.2, 10.8, 11.4,
                   12.0, 12.6, 13.2, 13.8, 14.4, 15.0, 15.6, 16.2, 16.8, 17.4])
h_agua = np.array([12.2, 11.8, 11.5, 11.1, 10.7, 10.3, 10.0, 9.6, 9.3, 9.0,
                   8.6, 8.3, 8.0, 7.6, 7.3, 7.0, 6.7, 6.5, 6.2, 5.9,
                   5.6, 5.4, 5.1, 4.9, 4.6, 4.4, 4.1, 3.9, 3.7, 3.5])

# ACEITE (26 mediciones, cada 0.7 s)
t_aceite = np.array([0.0, 0.7, 1.4, 2.1, 2.8, 3.5, 4.2, 4.9, 5.6, 6.3,
                     7.0, 7.7, 8.4, 9.1, 9.8, 10.5, 11.2, 11.9, 12.6, 13.3,
                     14.0, 14.7, 15.4, 16.1, 16.8, 17.5])
h_aceite = np.array([13.0, 12.7, 12.2, 11.8, 11.4, 10.9, 10.5, 10.2, 9.8, 9.4,
                     9.0, 8.6, 8.3, 8.0, 7.6, 7.3, 7.0, 6.7, 6.4, 6.1,
                     5.5, 5.2, 5.0, 4.2, 4.0, 3.7])

dh = 0.2
d1 = 5.987
dd1 = 0.2
d2 = 0.478
dd2 = 0.1
g = 9.8
dg = 0.05

# Punto 2: Areas y propagacion de error
A1 = np.pi * (d1 / 2.0) ** 2
dA1 = (np.pi / 2.0) * d1 * dd1
A2 = np.pi * (d2 / 2.0) ** 2
dA2 = (np.pi / 2.0) * d2 * dd2

print("==== AREAS (Punto 2) ====")
print("A1 [cm^2]:", A1, "+-", dA1)
print("A2 [cm^2]:", A2, "+-", dA2)
print()

def calcular_tf(h0_cm, d1_cm, d2_cm, g_ms2):
    h0_m = h0_cm / 100.0
    cociente = (d1_cm / d2_cm) ** 4      # A1^2 / A2^2
    return np.sqrt(2.0 * h0_m / g_ms2 * (cociente - 1.0))

def propagar_dtf(h0_val):
    signos = np.array([-1.0, 1.0])
    valores = np.empty(16)
    k = 0
    for a in signos:
        for b in signos:
            for c in signos:
                for e in signos:
                    valores[k] = calcular_tf(h0_val + a * dh, d1 + b * dd1,
                                             d2 + c * dd2, g + e * dg)
                    k = k + 1
    valores_ord = np.sort(valores)
    return (valores_ord[15] - valores_ord[0]) / 2.0

p = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
factor = 1.0 - np.sqrt(1.0 - p)

# ==========================================
# AGUA
# ==========================================
h0_agua = h_agua[0]
y_agua = h_agua / h0_agua
dy_agua = (dh / h0_agua) + (h_agua * dh) / (h0_agua ** 2)
er_agua = (dy_agua / y_agua) * 100.0

print("==== TABLA PUNTO 2: AGUA ====")
print("t [s] | h [cm] | dh [cm] | y(t) | dy(t) | er [%]")
cant_menor_5_agua = 0
for j in np.arange(0, t_agua.size, 1):
    if er_agua[j] < 5.0:
        cant_menor_5_agua = cant_menor_5_agua + 1
    print(t_agua[j], "|", h_agua[j], "|", dh, "|", y_agua[j], "|", dy_agua[j], "|", er_agua[j])
print("Mediciones con error relativo < 5%: ", cant_menor_5_agua, "de", t_agua.size)
print()

tf_agua = calcular_tf(h0_agua, d1, d2, g)
dtf_agua = propagar_dtf(h0_agua)

t_modelo_agua = tf_agua * factor
dt_modelo_agua = dtf_agua * factor

print("==== PUNTO 3: AGUA ====")
print("vaciado [%] | t modelo [s] +- dt [s] | t exp [s]")
for i in np.arange(0, p.size, 1):
    target_y = 1.0 - p[i]
    idx = 0
    while idx < y_agua.size and y_agua[idx] > target_y:
        idx = idx + 1
    if idx < y_agua.size:
        te = t_agua[idx-1] + (y_agua[idx-1] - target_y) / (y_agua[idx-1] - y_agua[idx]) * (t_agua[idx] - t_agua[idx-1])
        print(100.0 * p[i], "|", t_modelo_agua[i], "+-", dt_modelo_agua[i], "|", te)
    else:
        print(100.0 * p[i], "|", t_modelo_agua[i], "+-", dt_modelo_agua[i], "| fuera del rango medido (y_min =", y_agua[y_agua.size-1], ")")
print()


# ==========================================
# ACEITE
# ==========================================
h0_aceite = h_aceite[0]
y_aceite = h_aceite / h0_aceite
dy_aceite = (dh / h0_aceite) + (h_aceite * dh) / (h0_aceite ** 2)
er_aceite = (dy_aceite / y_aceite) * 100.0

print("==== TABLA PUNTO 2: ACEITE ====")
print("t [s] | h [cm] | dh [cm] | y(t) | dy(t) | er [%]")
cant_menor_5_aceite = 0
for j in np.arange(0, t_aceite.size, 1):
    if er_aceite[j] < 5.0:
        cant_menor_5_aceite = cant_menor_5_aceite + 1
    print(t_aceite[j], "|", h_aceite[j], "|", dh, "|", y_aceite[j], "|", dy_aceite[j], "|", er_aceite[j])
print("Mediciones con error relativo < 5%:", cant_menor_5_aceite, "de", t_aceite.size)
print()

tf_aceite = calcular_tf(h0_aceite, d1, d2, g)
dtf_aceite = propagar_dtf(h0_aceite)

t_modelo_aceite = tf_aceite * factor
dt_modelo_aceite = dtf_aceite * factor

print("==== PUNTO 3: ACEITE ====")
print("vaciado [%] | t modelo [s] +- dt [s] | t exp [s]")
for i in np.arange(0, p.size, 1):
    target_y = 1.0 - p[i]
    idx = 0
    while idx < y_aceite.size and y_aceite[idx] > target_y:
        idx = idx + 1
    if idx < y_aceite.size:
        te = t_aceite[idx-1] + (y_aceite[idx-1] - target_y) / (y_aceite[idx-1] - y_aceite[idx]) * (t_aceite[idx] - t_aceite[idx-1])
        print(100.0 * p[i], "|", t_modelo_aceite[i], "+-", dt_modelo_aceite[i], "|", te)
    else:
        print(100.0 * p[i], "|", t_modelo_aceite[i], "+-", dt_modelo_aceite[i], "| fuera del rango medido (y_min =", y_aceite[y_aceite.size-1], ")")
print()

