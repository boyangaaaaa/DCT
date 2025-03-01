# %%
import numpy as np
import time
from causallearn.utils.cit import CIT

def _binary(C, X_true):
    
    X = X_true.copy()
    i = 0
    for c in C:
        x_tmp = X[:,i]
        x_tmp[x_tmp > c] = 1
        x_tmp[x_tmp <= c] = 0
        X[:,i] = x_tmp
        i = i+1
    return X

x1 = np.random.normal(0,1,1000)
x2 = np.random.normal(0,1,1000) + 2 * x1 
x3 = np.random.normal(0,1,1000) + x1
x4 = np.random.normal(0,1,1000) + x2 + 2*x3 
x5 = np.random.normal(0,1,1000)
data = np.array([x1,x2,x3,x4, x5]).T
C = [np.random.uniform(-1,1) for i in range(5)]


X_obs = _binary(C, data)
test = 'dis_test'

start = time.time()
cit_test = CIT(X_obs, method=test)
cit_test(0,4)
end = time.time()
print(end - start)




# %%


# %%



