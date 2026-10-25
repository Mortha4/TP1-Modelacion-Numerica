import numpy as np


t = np.arange(0, 28, 1.0)
h = np.array([12.6, 12.0, 11.3, 10.7, 10.0, 9.5, 8.9, 8.3,
              7.8, 7.3, 6.8, 6.3, 5.9, 5.4, 5.0, 4.6,
              4.2, 3.4, 3.1, 2.8, 2.4, 2.2, 1.8, 1.7,
              1.4, 1.1, 0.9, 0.7])

h0 = h[0]       
dh = 0.1       
d1 = 5.987      
dd1 = 0.2       
d2 = 0.478      
dd2 = 0.1       
g = 9.8         
dg = 0.05       


y = h / h0

dy = (dh + y * dh) / h0
er_y = dy / y   # error relativo de y






def calcular_tf(h0_cm, d1_cm, d2_cm, g_ms2):
    h0_m = h0_cm / 100.0
    cociente = (d1_cm / d2_cm) ** 4      # A1^2 / A2^2
    return np.sqrt(2.0 * h0_m / g_ms2 * (cociente - 1.0))


tf = calcular_tf(h0, d1, d2, g)


signos = [-1.0, 1.0]
valores = []
for a in signos:
    for b in signos:
        for c in signos:
            for e in signos:
                valores.append(calcular_tf(h0 + a * dh, d1 + b * dd1,
                                           d2 + c * dd2, g + e * dg))
valores = np.array(valores)
dtf = (np.max(valores) - np.min(valores)) / 2.0




p = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
factor = 1.0 - np.sqrt(1.0 - p)
t_modelo = tf * factor
dt_modelo = dtf * factor    

print("vaciado | t modelo [s]")
for i in range(len(p)):
    print("%5.0f%%  | %5.1f +- %4.1f" % (100 * p[i], t_modelo[i], dt_modelo[i]))
