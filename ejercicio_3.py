import numpy as np


# AGUA
t_agua = np.arange(0, 28, 1.0)
h_agua = np.array([12.6, 12.0, 11.3, 10.7, 10.0, 9.5, 8.9, 8.3,
                   7.8, 7.3, 6.8, 6.3, 5.9, 5.4, 5.0, 4.6,
                   4.2, 3.4, 3.1, 2.8, 2.4, 2.2, 1.8, 1.7,
                   1.4, 1.1, 0.9, 0.7])

# ACEITE
t_aceite = np.arange(26, dtype=float) * 1.2
h_aceite = np.array([13.0, 12.2, 11.5, 10.8, 10.1,  9.4,  8.7,  8.1,
                      7.6,  7.0,  6.5,  6.0,  5.5,  5.0,  4.5,  4.1,
                      3.7,  3.3,  3.0,  2.7,  2.4,  2.1,  1.8,  1.6,
                      1.4,  1.2])

dh = 0.1       
d1 = 5.987      
dd1 = 0.2       
d2 = 0.478      
dd2 = 0.1       
g = 9.8         
dg = 0.05       

def calcular_tf(h0_cm, d1_cm, d2_cm, g_ms2):
    h0_m = h0_cm / 100.0
    cociente = (d1_cm / d2_cm) ** 4      # A1^2 / A2^2
    return np.sqrt(2.0 * h0_m / g_ms2 * (cociente - 1.0))

def propagar_dtf(h0_val):
    signos = [-1.0, 1.0]
    valores = []
    for a in signos:
        for b in signos:
            for c in signos:
                for e in signos:
                    valores.append(calcular_tf(h0_val + a * dh, d1 + b * dd1,
                                               d2 + c * dd2, g + e * dg))
    valores = np.array(valores)
    return (np.max(valores) - np.min(valores)) / 2.0

p = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
factor = 1.0 - np.sqrt(1.0 - p)

liquidos = [('AGUA', t_agua, h_agua), ('ACEITE', t_aceite, h_aceite)]

for nom, tt, hh in liquidos:
    h0 = hh[0]
    tf = calcular_tf(h0, d1, d2, g)
    dtf = propagar_dtf(h0)

    t_modelo = tf * factor
    dt_modelo = dtf * factor    

    print(f"==== {nom} ====")
    print("vaciado | t modelo [s] | t exp [s]")
    for i in range(len(p)):
        target_y = 1.0 - p[i]
        idx = 0
        y = hh / h0
        while idx < len(y) and y[idx] > target_y:
            idx += 1
        te = tt[idx-1] + (y[idx-1] - target_y) / (y[idx-1] - y[idx]) * (tt[idx] - tt[idx-1])
        print("%5.0f%%  | %5.1f +- %4.1f | %7.2f" % (100 * p[i], t_modelo[i], dt_modelo[i], te))
    print()
