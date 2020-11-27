import numpy as np
def newton(f,Df,x0,epsilon,max_iter):
    xn = x0
    
    for n in range(0,max_iter):
        fxn = f(xn)
    
        if abs(fxn) < epsilon:
            print('Found solution after',n,'iterations.')
            return xn
    
        Dfxn = Df(xn)
    
        if Dfxn == 0:
            print('Zero derivative. No solution found.')
            return None
    
        xn = xn - fxn/Dfxn
    
    print('Exceeded maximum iterations. No solution found.')
    
    return None

def f(x):
    return x*np.exp(-2*(x-1)) - 1
def Df(x):
    return np.exp(-2*(x-1))-2*x*(x-1)*np.exp(-2*(x-1))
approx = newton(f,Df,0.001,1e-15,10000)
print(approx)