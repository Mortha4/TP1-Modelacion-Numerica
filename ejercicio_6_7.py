import numpy as np

d1 = 5.987
dd1 = 0.2
d2 = 0.478
dd2 = 0.1
g = 9.8
dg = 0.05
dh = 0.2

def calcular_tf(h0_cm, d1_cm, d2_cm, g_ms2):
    return np.sqrt(2.0 * (h0_cm / 100.0) / g_ms2 * ((d1_cm / d2_cm) ** 4 - 1.0))

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

obj = np.array([0.50, 0.10])

# ==========================================
# AGUA
# ==========================================
print('==== AGUA ====')
h0_agua = h_agua[0]
y_agua = h_agua / h0_agua
dy_agua = (dh / h0_agua) + (h_agua * dh) / (h0_agua ** 2)
er_agua = dy_agua / y_agua

tf_agua = calcular_tf(h0_agua, d1, d2, g)
dtf_agua = propagar_dtf(h0_agua)

sel_agua = er_agua < 0.10
ta_agua = t_agua[sel_agua]
ya_agua = y_agua[sel_agua]

phi0_agua = np.ones(ta_agua.size)
phi1_agua = ta_agua
phi2_agua = ta_agua ** 2
phi3_agua = ta_agua ** 3

M3_agua = np.array([
    [np.sum(phi0_agua * phi0_agua), np.sum(phi0_agua * phi1_agua), np.sum(phi0_agua * phi2_agua), np.sum(phi0_agua * phi3_agua)],
    [np.sum(phi1_agua * phi0_agua), np.sum(phi1_agua * phi1_agua), np.sum(phi1_agua * phi2_agua), np.sum(phi1_agua * phi3_agua)],
    [np.sum(phi2_agua * phi0_agua), np.sum(phi2_agua * phi1_agua), np.sum(phi2_agua * phi2_agua), np.sum(phi2_agua * phi3_agua)],
    [np.sum(phi3_agua * phi0_agua), np.sum(phi3_agua * phi1_agua), np.sum(phi3_agua * phi2_agua), np.sum(phi3_agua * phi3_agua)]
])
b3_agua = np.array([np.sum(ya_agua * phi0_agua), np.sum(ya_agua * phi1_agua), np.sum(ya_agua * phi2_agua), np.sum(ya_agua * phi3_agua)])
aC_agua, bC_agua, cC_agua, dC_agua = np.linalg.solve(M3_agua, b3_agua)

for o in obj:
    def f(x):
        return aC_agua + bC_agua * x + cC_agua * (x ** 2) + dC_agua * (x ** 3) - o

    def df(x):
        return bC_agua + 2.0 * cC_agua * x + 3.0 * dC_agua * (x ** 2)

    x0 = 10.0  # la semilla
    tol = 0.5e-9
    i = 1
    x1 = x0 - f(x0) / df(x0)
    while abs(x1 - x0) > tol:
        i = i + 1
        x0 = x1
        x1 = x0 - f(x0) / df(x0)

    tt = tf_agua * (1.0 - np.sqrt(o))
    dtt = dtf_agua * (1.0 - np.sqrt(o))

    print('objetivo y =', o, '(vaciado del', (1.0 - o) * 100.0, '%)')
    print('  ec. (1) y (2) :', tt, '+-', dtt)
    print('  cubico (N-R)  :', x1, '(iteraciones:', i, ', |x1 - x0| =', abs(x1 - x0), ')')
    print('  diferencia    :', x1 - tt)

    idx = 0
    while idx < y_agua.size and y_agua[idx] > o:
        idx = idx + 1
    if idx < y_agua.size:
        te = t_agua[idx-1] + (y_agua[idx-1] - o) / (y_agua[idx-1] - y_agua[idx]) * (t_agua[idx] - t_agua[idx-1])
        print('  experimental  :', te)
    else:
        print('  experimental  : fuera del rango medido (y_min =', y_agua[y_agua.size-1], ')')
    print()


# ==========================================
# ACEITE
# ==========================================
print('==== ACEITE ====')
h0_aceite = h_aceite[0]
y_aceite = h_aceite / h0_aceite
dy_aceite = (dh / h0_aceite) + (h_aceite * dh) / (h0_aceite ** 2)
er_aceite = dy_aceite / y_aceite

tf_aceite = calcular_tf(h0_aceite, d1, d2, g)
dtf_aceite = propagar_dtf(h0_aceite)

sel_aceite = er_aceite < 0.10
ta_aceite = t_aceite[sel_aceite]
ya_aceite = y_aceite[sel_aceite]

phi0_aceite = np.ones(ta_aceite.size)
phi1_aceite = ta_aceite
phi2_aceite = ta_aceite ** 2
phi3_aceite = ta_aceite ** 3

M3_aceite = np.array([
    [np.sum(phi0_aceite * phi0_aceite), np.sum(phi0_aceite * phi1_aceite), np.sum(phi0_aceite * phi2_aceite), np.sum(phi0_aceite * phi3_aceite)],
    [np.sum(phi1_aceite * phi0_aceite), np.sum(phi1_aceite * phi1_aceite), np.sum(phi1_aceite * phi2_aceite), np.sum(phi1_aceite * phi3_aceite)],
    [np.sum(phi2_aceite * phi0_aceite), np.sum(phi2_aceite * phi1_aceite), np.sum(phi2_aceite * phi2_aceite), np.sum(phi2_aceite * phi3_aceite)],
    [np.sum(phi3_aceite * phi0_aceite), np.sum(phi3_aceite * phi1_aceite), np.sum(phi3_aceite * phi2_aceite), np.sum(phi3_aceite * phi3_aceite)]
])
b3_aceite = np.array([np.sum(ya_aceite * phi0_aceite), np.sum(ya_aceite * phi1_aceite), np.sum(ya_aceite * phi2_aceite), np.sum(ya_aceite * phi3_aceite)])
aC_aceite, bC_aceite, cC_aceite, dC_aceite = np.linalg.solve(M3_aceite, b3_aceite)

for o in obj:
    def f(x):
        return aC_aceite + bC_aceite * x + cC_aceite * (x ** 2) + dC_aceite * (x ** 3) - o

    def df(x):
        return bC_aceite + 2.0 * cC_aceite * x + 3.0 * dC_aceite * (x ** 2)

    x0 = 10.0  # la semilla
    tol = 0.5e-9
    i = 1
    x1 = x0 - f(x0) / df(x0)
    while abs(x1 - x0) > tol:
        i = i + 1
        x0 = x1
        x1 = x0 - f(x0) / df(x0)

    tt = tf_aceite * (1.0 - np.sqrt(o))
    dtt = dtf_aceite * (1.0 - np.sqrt(o))

    print('objetivo y =', o, '(vaciado del', (1.0 - o) * 100.0, '%)')
    print('  ec. (1) y (2) :', tt, '+-', dtt)
    print('  cubico (N-R)  :', x1, '(iteraciones:', i, ', |x1 - x0| =', abs(x1 - x0), ')')
    print('  diferencia    :', x1 - tt)

    idx = 0
    while idx < y_aceite.size and y_aceite[idx] > o:
        idx = idx + 1
    if idx < y_aceite.size:
        te = t_aceite[idx-1] + (y_aceite[idx-1] - o) / (y_aceite[idx-1] - y_aceite[idx]) * (t_aceite[idx] - t_aceite[idx-1])
        print('  experimental  :', te)
    else:
        print('  experimental  : fuera del rango medido (y_min =', y_aceite[y_aceite.size-1], ')')
    print()