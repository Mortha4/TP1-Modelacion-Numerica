import numpy as np


t = np.arange(0, 28, 1.0)
h = np.array([12.575, 11.951, 11.264, 10.665, 10.022, 9.451, 8.907, 8.31,
              7.781, 7.283, 6.785, 6.287, 5.857, 5.374, 4.954, 4.56,
              4.181, 3.434, 3.107, 2.76, 2.438, 2.168, 1.816, 1.66,
              1.421, 1.131, 0.949, 0.737])

h0 = h[0]          
dh = 0.1          
d1 = 6.0           
dd1 = 0.2         
d2 = 0.5           
dd2 = 0.1          
g = 9.8           
dg = 0.05          

y = h / h0
dy = (dh + y * dh) / h0     # cota de y = h/h0 (con dh0 = dh)

#calcular tiempo final 
def calcular_tf(h0_cm, d1_cm, d2_cm, g_ms2):
    h0_m = h0_cm / 100.0

    cociente = (d1_cm / d2_cm) ** 4
    return np.sqrt(2.0 * h0_m / g_ms2 * (cociente - 1.0))


tf = calcular_tf(h0, d1, d2, g)


signos = [-1.0, 1.0]
valores = []

#calcula el valor de tiempo teorico usando el tiempo final
for a in signos:
    for b in signos:
        for c in signos:
            for e in signos:
                valores.append(calcular_tf(h0 + a * dh, d1 + b * dd1,
                                           d2 + c * dd2, g + e * dg))
valores = np.array(valores)
tf_min = np.min(valores)
tf_max = np.max(valores)
dtf = (tf_max - tf_min) / 2.0


#porcentajes de tiempo de vaciado 
p = np.array([0.10, 0.25, 0.50, 0.75, 0.90])
factor = 1.0 - np.sqrt(1.0 - p)
t_modelo = tf * factor    
dt_modelo = dtf * factor     



def tiempo_experimental(y_obj):

    for i in range(len(y) - 1):
        if y[i] >= y_obj >= y[i + 1]:
            pendiente = (y[i] - y[i + 1]) / (t[i + 1] - t[i])   # |dy/dt|
            t_exp = t[i] + (y[i] - y_obj) / pendiente
            cota_y = (dh + y_obj * dh) / h0
            return t_exp, cota_y / pendiente
    return np.nan, np.nan








print("vaciado | t modelo [s]")
for i in range(len(p)):
    print("%5.0f%%  | %5.1f +- %4.1f" % (100 * p[i], t_modelo[i], dt_modelo[i]))