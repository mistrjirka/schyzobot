import numpy as np

def fit_functions(funcs, data):
    X = [np.arange(len(data)) for _ in range(len(funcs))]
    
    Y = []
    for func in funcs:
        y_vals = list(map(lambda x: func(x), X[0]))
        Y.append(y_vals)
        
    A = np.array(Y).T
    B = data
    
    coefficients, _,_,_  = np.linalg.lstsq(A,B,rcond=None)

    return [lambda x : a + b*x for (a,b) in zip([0]+coefficients[:-1],coefficients[1:])]  

# Test the function:
funcs=[lambda x:2*x**3+5*x**2-7*x+10, lambda x:-x]
data = np.random.rand(100)
print(fit_functions(funcs,data)) 