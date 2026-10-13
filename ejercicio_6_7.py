import numpy as np

aC=0.99618042
bC=-0.05023185
cC=0.00037686
dC=0.0000071893

tf=25.131

t=np.arange(28,dtype=float)
h=np.array([12.575,11.951,11.264,10.665,10.022,9.451,8.907,8.310,7.781,7.283,6.785,6.287,5.857,5.374,4.954,4.560,4.181,3.434,3.107,2.760,2.438,2.168,1.816,1.660,1.421,1.131,0.949,0.737])
y=h/h[0]

obj=[0.50,0.10]

for o in obj:

    f=lambda x: aC+bC*x+cC*x**2+dC*x**3-o
    df=lambda x: bC+2*cC*x+3*dC*x**2

    x=10.0
    tol=0.5e-9
    n=0
    x1=x-f(x)/df(x)
    while abs(x1-x)>tol:
        x=x1
        x1=x-f(x)/df(x)
        n=n+1

    tt=tf*(1-np.sqrt(o))

    i=0
    while y[i]>o:
        i=i+1
    te=t[i-1]+(y[i-1]-o)/(y[i-1]-y[i])*(t[i]-t[i-1])

    print('objetivo y =',o)
    print('  ec. (1) y (2)  ',tt)
    print('  cubico        ',x1)
    print('  experimental  ',te)
    print('  diferencia    ',x1-tt,'   iteraciones',n)
    print()