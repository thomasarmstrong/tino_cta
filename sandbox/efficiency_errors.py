from math import factorial, log, exp, gamma, pi
import numpy as np
from scipy.optimize import minimize

from itertools import count

__all__ = [
            "get_efficiency_errors",
            "get_efficiency_errors_minimize",
            "get_efficiency_errors_scan",
            ]

#gamma = lambda x: factorial(x-1)
def log_gamma(x):
    # Gamma(x) = (x-1)!
    return sum( log(i) for i in np.arange(1,x) )
    
    
def P_e (k, N, e):
    ''' has problems with high numbers, use log-version instead '''
    #res1 = gamma(N+2) / (gamma(k+1)*gamma(N-k+1)) * e**k * (1-e)**(N-k)
    #return res1 / 6
    
    ''' protect from log(0) calls '''
    if e == 0:
        if k == 0: return 1
        else:      return 0
    if e == 1:
        if k == N: return 1
        else:      return 0
    
    res2 = log_gamma(N+2) + k*log(e) + (N-k)*log(1-e) - (log_gamma(k+1) + log_gamma(N-k+1))
    return exp(res2)


def test_func(arg,k,N, conf=.68, func=P_e,de=.001):
    # test function that calculates the 
    # difference between a and b while covering an intervall of conf
    a = arg[0]
    print(a)
    b = a
    integral=0
    for i in count():
        integral += (func(k,N,b) + func(k,N,b+de))*de /2.
        b = a + i*de
        if integral >= .68: 
            test_func.a = a
            test_func.b = b
            return b-a
        if b > 1 :          return 1.1
    
def test_func_2(a,k,N, conf=.68, func=P_e,de=.001):
    # test function that calculates the 
    # difference between a and b while covering an intervall of conf
    # use in scan mode, returns b instead of b-a
    b = a
    integral=0
    for i in count():
        integral += (func(k,N,b) + func(k,N,b+de))*de /2.
        b = a + i*de
        if integral >= .68: return (b,integral)
        if b+de > 1 :       return (9,integral)

def get_efficiency_errors_minimize(k, N, conf=.68) :
    minimize(test_func, [0], args=(k,N), bounds=[(0,1)],
             method='L-BFGS-B', options={'disp' : False, 'eps':1e-10}
            )
    return [k/N, test_func.a, test_func.b]

def get_efficiency_errors_scan(k, N, conf=.68) :
    
    if N == 0: return [0,0,0,0,0]
    
    de = 0.0005
    min_diff = 20.
    min_a = None
    min_b = None
    for i in count(0):
        a = i * de
        (b,integral) = test_func_2(a, k, N)
        if b-a < min_diff: 
            min_diff = b-a
            min_a = a
            min_b = b
        if 1 < b: break
    
    
    mean = k/N 
    lerr = mean-min_a
    herr = min_b-mean
    if k == 0: lerr = 0
    if k == N: herr = 1
    
    return [mean, lerr, herr, min_a, min_b, integral]


get_efficiency_errors = get_efficiency_errors_scan




print(get_efficiency_errors(10,50)) 

'''
import numpy as np
import matplotlib.pyplot as plt

dt = 0.001
t = np.arange(0.0, 1.+dt, dt)
y0 = [P_e(0,5,x) for x in t]
y1 = [P_e(1,5,x) for x in t]
y2 = [P_e(2,5,x) for x in t]
y3 = [P_e(3,5,x) for x in t]
y4 = [P_e(4,5,x) for x in t]
y5 = [P_e(5,5,x) for x in t]




print( sum(y0) * dt )
print( sum(y1) * dt )
print( sum(y2) * dt )
print( sum(y3) * dt )
print( sum(y4) * dt )
print( sum(y5) * dt )




plt.figure(1)
plt.plot(y0, 'bo', y1, 'ro', y2, 'yo', y3, 'go', y4, "bo", y5, "ro")
plt.show()

'''