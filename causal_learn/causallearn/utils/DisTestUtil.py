import numpy as np
from joblib import Parallel, delayed
from scipy import stats
from scipy.optimize import fminbound
from scipy.stats import mvn
import sympy as sp
from multiprocessing import Pool
from concurrent.futures import ThreadPoolExecutor

    
def determine_type(X):
        
    cardinalities = []
    for col_idx in range(X.shape[1]):
        unique_vals = np.unique(X[:, col_idx])
        cardinalities.append(len(unique_vals))
    
    # Find all dimension indices with cardinality <= 5
    all_le_5 = all(c <= 5 for c in cardinalities)
    all_gt_5 = all(c > 5 for c in cardinalities)

    if all_le_5:
        data_type = "binary"
        # Binarize every column
        X_output = _binary_based_mean(X, np.arange(X.shape[1]))
    elif all_gt_5:
        data_type = "continuous"
        print("The data is continuous !!! You should not use DCT at all !!!")
        # No binarization - return a copy of the original
        X_output = X.copy()
    else:
        dims_le_5 = [idx for idx, c in enumerate(cardinalities) if c <= 5]        
        data_type = "mixed"
        # Binarize only columns with cardinality <= 5
        X_output = _binary_based_mean(X, dims_le_5)

    return data_type, X_output


def choose_Psi(X, j1, j2 ):
    if len(set(X[:,j1])) == 2 and len(set(X[:,j2])) == 2:
        return Psi_vector(X, j1, j2)
    elif len(set(X[:,j1])) > 2 and len(set(X[:,j2])) == 2:
        return Psi_vector_mixed(X, j1, j2)
    elif len(set(X[:,j1])) == 2 and len(set(X[:,j2])) > 2:
        return Psi_vector_mixed(X, j2, j1)
    else:
        return Psi_con(X, j1, j2)


def choose_Psi_grad(X, j1, j2):
    if len(set(X[:,j1])) == 2 and len(set(X[:,j2])) == 2:
        return Psi_grad(X, j1, j2)
    elif len(set(X[:,j1])) > 2 and len(set(X[:,j2])) == 2:
        return Psi_grad_mixed(X, j1, j2)
    elif len(set(X[:,j1])) == 2 and len(set(X[:,j2])) > 2:
        return Psi_grad_mixed(X, j2, j1)
    else:
        return Psi_con(X, j1, j2)


def _binary_based_mean(X, indexs):
    for i in indexs:
        x_tmp = X[:,i]
        c = np.mean(x_tmp)
        x_tmp[x_tmp < c] = 0
        x_tmp[x_tmp >= c] = 1
        X[:, i] = x_tmp        

    return X 


def _trigger_stable(X, j1, j2):
    if len(set(X[:,j1])) == 2 and len(set(X[:,j2])) == 2:
        return False
    else:
        return True


class Estimation():
    def __init__(self, X_obs):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]

    def _calc_h_hat(self,j):
        X = self.X
        ratio = 1 - (X[:,j].sum()/self.n)
        h_hat = stats.norm.ppf(ratio)

        return h_hat

    def _calc_pi_hat_j(self,j):
        X = self.X
        ratio = X[:,j].sum()/self.n
        pi_hat_j = ratio

        return pi_hat_j

    def _calc_pi_hat(self,j1,j2):
        X = self.X
        ratio = (X[:,j1]*X[:,j2]).sum()/self.n
        pi_hat = ratio

        return pi_hat

    # where j1 is the index of continuous, j2 is the index of binary
    def _calc_pi_hat_mixed(self, j1, j2):
        X = self.X 
        ratio = sum(1 for xy in X if xy[j1]>=0 and xy[j2]==1)/self.n
        pi_hat = ratio

        return pi_hat


    def calc_pi_hat_func(self, j1,j2, sigma):
        h_hat_j1 = self._calc_h_hat(j1)
        h_hat_j2 = self._calc_h_hat(j2)
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)
        mean = [0, 0]  # Means for x and y
        # Upper limits are infinity (representing the tail of the distribution)
        upper_limit = np.array([np.inf, np.inf])
        # Lower limits
        lower_limit = np.array([h_hat_j1, h_hat_j2])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        pi_hat = prob

        return pi_hat


    # j2 is the index of the binary variable
    def calc_pi_hat_func_mixed(self, j2, sigma):
        h_hat_j1 = 0 
        h_hat_j2 = self._calc_h_hat(j2)
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)
        mean = [0, 0]  # Means for x and y
        # Upper limits are infinity (representing the tail of the distribution)
        upper_limit = np.array([np.inf, np.inf])
        # Lower limits
        lower_limit = np.array([h_hat_j1, h_hat_j2])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        pi_hat = prob

        return pi_hat        
        

    def _calc_sigma_hat(self, j1, j2):
        # determine X_j1 is continous or not
        if len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) == 2:
            pi_hat = self._calc_pi_hat(j1, j2)
            obj = lambda sigma_hat: (self.calc_pi_hat_func(j1,j2, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        elif len(set(self.X[:,j2])) == 2 and len(set(self.X[:,j1])) > 2:
            pi_hat = self._calc_pi_hat_mixed(j1, j2)
            obj = lambda sigma_hat: (self.calc_pi_hat_func_mixed(j2, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        elif len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) > 2:
            pi_hat = self._calc_pi_hat_mixed(j2, j1)
            obj = lambda sigma_hat: (self.calc_pi_hat_func_mixed(j1, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        # covariance of two continous variables
        else: 
            sigma_hat = np.corrcoef(self.X[:,j1], self.X[:,j2])[0,1]

        return sigma_hat


    def _calc_all_sigma(self): # return all sigma_hat as a matrix
        p = self.p
        sigma_hat_mat = np.zeros((p,p))
        # Parallel to find out the sigma_hat_matrix and return as a matrix
        # sigma_hat_mat = Parallel(n_jobs=-1)(delayed(self._calc_sigma_hat)(j1,j2) for j1 in range(p) for j2 in range(p))
        arg_list = [(j1, j2) for j1 in range(p) for j2 in range(p)]
        with ThreadPoolExecutor(max_workers=4) as executor:
            sigma_hat_mat = list(executor.map(lambda args: self._calc_sigma_hat(*args), arg_list))

        sigma_hat_mat = np.array(sigma_hat_mat).reshape(p,p)
        
        return sigma_hat_mat

    
    def _calc_sigma_hat_j(self, j):
        sigma_hat_j = np.zeros((self.p-1,1))
        # sigma_hat_j = Parallel(n_jobs=-1)(delayed(self._calc_sigma_hat)(j,j1) for j1 in range(self.p) if j1 != j) 
        arg_list = [(j, j1) for j1 in range(self.p) if j1 != j]
        with ThreadPoolExecutor(max_workers=4) as executor:
            sigma_hat_j = list(executor.map(lambda args: self._calc_sigma_hat(*args), arg_list))

        return np.array(sigma_hat_j).reshape(-1,1)

class Psi_vector():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        self.pi_hat_12 = self.estimation._calc_pi_hat(j1,j2)
        self.pi_hat1 = self.estimation._calc_pi_hat_j(j1)
        self.pi_hat2 = self.estimation._calc_pi_hat_j(j2)

    def Psi_theta_1(self, i):
        X = self.X
        j1 = self.j1 
        j2 = self.j2
        pi_hat_12_i = X[i,j1] * X[i,j2]

        psi_theta_1 = pi_hat_12_i - self.pi_hat_12

        return psi_theta_1

    def Psi_theta_2(self, i):
        X = self.X
        j1 = self.j1
        pi_hat1  = X[i, j1] - self.pi_hat1

        return pi_hat1

    def Psi_theta_3(self, i):
        X = self.X
        j2 = self.j2
        pi_hat2  = X[i, j2] - self.pi_hat2

        return pi_hat2    
    
    def Psi_theta_vector(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)
        psi_theta3 = self.Psi_theta_3(i)

        return np.array([psi_theta1, psi_theta2, psi_theta3]).reshape(-1,1)

    # calculate the matrix Psi * Psi.T for ith entry
    def Psi_theta_matrix(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)
        psi_theta3 = self.Psi_theta_3(i)

        psi_theta_vector = np.array([psi_theta1, psi_theta2, psi_theta3]).reshape(-1,1)

        psi_theta_matrix = psi_theta_vector @ psi_theta_vector.T

        return psi_theta_matrix
    
    
class Psi_grad():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        estimation = self.estimation
        self.pi_hat_12 = estimation._calc_pi_hat(j1,j2)
        self.pi_hat1 = estimation._calc_pi_hat_j(j1)
        self.pi_hat2 = estimation._calc_pi_hat_j(j2)
        self.h_hat1 = estimation._calc_h_hat(j1)
        self.h_hat2 = estimation._calc_h_hat(j2)
        self.sigma_hat = estimation._calc_sigma_hat(j1, j2)

    def Psi_11(self):
        h_hat1 = self.h_hat1
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = np.exp(-(h_hat1**2 - 2*sigma_hat*h_hat1 * h_hat2 + h_hat2**2)/(2*(1- sigma_hat**2)))
        denominator = -2*np.pi * np.sqrt(1-sigma_hat**2)

        return numerator/denominator
    
    def Psi_12(self):
        x = sp.symbols('x')
        h_hat1 = self.h_hat1
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = sp.exp(-(h_hat1**2 - 2*sigma_hat*h_hat1 * x + x**2)/(2*(1- sigma_hat**2)))
        denominator = -2*sp.pi * sp.sqrt(1-sigma_hat**2)
        f = numerator/denominator

        definite_integral = sp.integrate(f,(x, h_hat2, sp.oo))
        
        return -definite_integral


    def Psi_13(self):
        x = sp.symbols('x')
        h_hat1 = self.h_hat1
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = sp.exp(-(h_hat2**2 - 2*sigma_hat*h_hat2 * x + x**2)/(2*(1- sigma_hat**2)))
        denominator = -2*sp.pi * sp.sqrt(1-sigma_hat**2)
        f = numerator/denominator

        definite_integral = sp.integrate(f,(x, h_hat1, sp.oo))

        return -definite_integral
        # return 0
        
    def Psi_21(self):

        return 0
    
    def Psi_22(self):
        h_hat1 = self.h_hat1

        return np.exp(- h_hat1**2 / 2)/np.sqrt(2 * np.pi)
    
    def Psi_23(self):

        return 0
    
    def Psi_31(self):

        return 0
    
    def Psi_32(self):

        return 0
    
    def Psi_33(self):
        h_hat2 = self.h_hat2

        return np.exp(- h_hat2**2 / 2)/np.sqrt(2 * np.pi)
    
    def Psi_grad_matrix(self):
        
        return np.array([
            [self.Psi_11(), self.Psi_12(), self.Psi_13()],
            [self.Psi_21(), self.Psi_22(), self.Psi_23()],
            [self.Psi_31(), self.Psi_32(), self.Psi_33()]
        ]).astype(np.float64)

        
# j1 is the index of the continous variable j2 is the index of binary variable
class Psi_vector_mixed():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        self.pi_hat_12 = self.estimation._calc_pi_hat_mixed(j1,j2)
        self.pi_hat2 = self.estimation._calc_pi_hat_j(j2)   

    def Psi_theta_1(self, i):
        X = self.X
        j1 = self.j1
        j2 = self.j2
        # return 1 if X[i,j1] > 0 and X[i,j2] == 1 else 0
        pi_hat_12_i = 1 if X[i,j1] >= 0 and X[i,j2] == 1 else 0
        psi_hat_12_i = pi_hat_12_i - self.pi_hat_12

        return psi_hat_12_i

    def Psi_theta_2(self, i):
        X = self.X
        j2 = self.j2
        pi_hat2  = X[i, j2] - self.pi_hat2

        return pi_hat2

    def Psi_theta_vector(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)

        return np.array([psi_theta1, psi_theta2]).reshape(-1,1)

    # calculate the matrix Psi * Psi.T for ith entry
    def Psi_theta_matrix(self, i):
        psi_theta1 = self.Psi_theta_1(i)
        psi_theta2 = self.Psi_theta_2(i)

        psi_theta_vector = np.array([psi_theta1, psi_theta2]).reshape(-1,1)

        psi_theta_matrix = psi_theta_vector @ psi_theta_vector.T

        return psi_theta_matrix


# j1 is continuous, j2 is binary
class Psi_grad_mixed():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.estimation = Estimation(self.X)
        estimation = self.estimation
        self.pi_hat_12 = estimation._calc_pi_hat_mixed(j1,j2)
        self.pi_hat2 = estimation._calc_pi_hat_j(j2)
        self.h_hat2 = estimation._calc_h_hat(j2)
        self.sigma_hat = estimation._calc_sigma_hat(j1, j2)

    def Psi_11(self):
        h_hat1 = 0
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = np.exp(-(h_hat1**2 - 2*sigma_hat*h_hat1 * h_hat2 + h_hat2**2)/(2*(1- sigma_hat**2)))
        denominator = -2*np.pi * np.sqrt(1-sigma_hat**2)

        return numerator/denominator
    
    def Psi_12(self):
        x = sp.symbols('x')
        h_hat1 = 0
        h_hat2 = self.h_hat2
        sigma_hat = self.sigma_hat
        numerator = sp.exp(-(h_hat2**2 - 2*sigma_hat*h_hat2 * x + x**2)/(2*(1- sigma_hat**2)))
        denominator = -2*sp.pi * sp.sqrt(1-sigma_hat**2)
        f = numerator/denominator

        definite_integral = sp.integrate(f,(x, h_hat1, sp.oo))

        return -definite_integral
    
    def Psi_21(self):

        return 0
    
    def Psi_22(self):
        h_hat2 = self.h_hat2

        return np.exp(- h_hat2**2/2)/np.sqrt(2 * np.pi)

    def Psi_grad_matrix(self):
        
        return np.array([
            [self.Psi_11(), self.Psi_12()],
            [self.Psi_21(), self.Psi_22()],
        ]).astype(np.float64)
        

# Continuous case
class Psi_con():
    def __init__(self, X_obs, j1, j2):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.j1 = j1
        self.j2 = j2
        self.j1mean = self.X[:,j1].mean()
        self.j2mean = self.X[:,j2].mean()
        self.estimation = Estimation(self.X)
        self.sigma_hat = np.corrcoef(self.X[:,j1], self.X[:,j2])[0,1]


    def Psi_theta_vector(self, i):
        xj1 = self.X[i, self.j1]
        xj2 = self.X[i, self.j2]
        psi_theta_1 = xj1 * xj2 - self.j1mean * self.j2mean - self.sigma_hat
        
        return psi_theta_1.reshape(1,1)


    def Psi_theta_matrix(self, i):
        xj1 = self.X[i, self.j1]
        xj2 = self.X[i, self.j2]
        psi_theta_1 = (xj1 * xj2 - self.j1mean * self.j2mean - self.sigma_hat) **2
        
        return psi_theta_1.reshape(1,1)


    def Psi_grad_matrix(self):
        
        return np.array([-1]).reshape(1,1)

# gradient will be 1
    def Var(self):
        tmp = 0
        for i in range(self.n):
            tmp = tmp + self.Psi_theta_vector(i)**2
            
        return tmp/self.n
        

class disTest_binary():
    def __init__(self, X_obs):

        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.index_mapping = np.arange(self.p*self.p).reshape(self.p, self.p)
    
        self.estimation = Estimation_binary(self.X)
        self.sigma_hat_mat = self.estimation.sigma_hat_all
        self.grad_all, self.grad_all_inv = self.estimation.grad_all, self.estimation.grad_all_inv 
        self.indexs = self._save_indexs()   # indexs of zero and one for each j1, j2
        self.look_up_tables = self.look_up_table_all()
        self.xai_all = self._save_all_xai_i()
        
   
    def look_up_table(self, j1, j2):
        look_up_dict = {}
        for idx, i in enumerate(self.indexs[j1][j2]):
            for number in i:
                look_up_dict[number] = idx
                
        return look_up_dict
    
    
    def look_up_table_all(self):
        look_up_table_all = []
        for j1 in range(self.p):
            look_up_table_row = []
            for j2 in range(self.p):
                look_up_table_row.append(self.look_up_table(j1, j2))
            look_up_table_all.append(look_up_table_row) 
        
        return look_up_table_all
    
        
    def get_varaince(self, X, j1, j2):
        X = X.copy()
        n = X.shape[0]
        psi_vector = choose_Psi(X, j1, j2)
        psi_matrix_summation = np.zeros_like(psi_vector.Psi_theta_matrix(0))
        for i in range(n):
            psi_matrix = psi_vector.Psi_theta_matrix(i)
            psi_matrix_summation = psi_matrix + psi_matrix_summation
        psi_matrix_average = psi_matrix_summation/n    
        psi_grad = choose_Psi_grad(X, j1, j2)
        psi_grad_matrix = psi_grad.Psi_grad_matrix()
        variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)

        return variance_vector[0,0]


    # return a p-1xp-1 matrix using parallel computing
    def get_varaince_all(self, j):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        X_tmp = np.delete(X_tmp, j, axis=1) 
        variance_all = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p-1) for j2 in range(p-1))
        variance_all = np.array(variance_all).reshape(p-1, p-1)
        
        return variance_all


    def get_varaince_whole_mat(self):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        variance_mat = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p) for j2 in range(p))
        variance_mat = np.array(variance_mat).reshape(p, p)

        return variance_mat


    # return a p-1 vector using parallel computing
    def get_varaince_vector(self, j):
        X = self.X.copy()
        p = X.shape[1]
        variance_vector = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self.get_varaince)(X, j1, j) for j1 in range(p) if j1 != j)
        variance_vector = np.array(variance_vector).reshape(p-1,1)

        return variance_vector


    def _calc_sigma_mat_minus_j(self, j): 
        # return the sigma matrix without the jth row and column
        return np.delete(np.delete(self.sigma_hat_mat, j, 0), j, 1)
    

    def _calc_sigma_mat_minus_j_inv(self,j):

        return np.linalg.inv(self._calc_sigma_mat_minus_j(j))
    
    
    def _calc_sigma_vector_minusj_j(self, j):

        return self.estimation._calc_sigma_hat_j(j)
    

    def _calc_xai_i(self, X, i, j1, j2):
        p = self.p
        psi_vector_ins = choose_Psi(X, j1, j2)
        psi_vector = psi_vector_ins.Psi_theta_vector(i)
        # if j1> j2:
        #     j1, j2 = j2, j1
        psi_grad_inv = self.grad_all_inv[self.index_mapping[j1, j2]]
        # psi_grad_inv = self.grad_all_inv[j1, j2]
        xai_i = psi_grad_inv @ psi_vector

        return xai_i[0] 
    

    def _obtain_xai_i(self, i, j1, j2):
        i_belongs_to = self.look_up_tables[j1][j2][i]
        xai_i = self.xai_all[j1][j2][i_belongs_to]
        
        return xai_i
    
    
    def _obtain_xai_all_minusj_minusj_i(self, i, j):
        xai_all = np.zeros((self.p, self.p))
        for j1 in range(self.p):
            for j2 in range(self.p):
                xai_all[j1, j2] = self._obtain_xai_i(i, j1, j2)
                
        xai_all_minusj = np.delete(np.delete(xai_all, j, axis=1), j, axis=0)
        
        return xai_all_minusj
    
    
    def _obtain_xai_all_minusj_j_i(self, i, j):
        xai_all = np.zeros((self.p, 1))
        for j1 in range(self.p):
            xai_all[j1] = self._obtain_xai_i(i, j1, j)
        xai_all = np.delete(xai_all, j, axis=0)
        
        return xai_all    
    

    def _save_index(self, j1, j2):
        indexs = []
        unique_rows = {tuple(row) for row in self.X[:,[j1, j2]]}
        combinations = [np.array(row) for row in unique_rows]
        for i in range(len(combinations)):
            condition = np.all(self.X[:,[j1, j2]] == combinations[i], axis=1)
            indexs.append(np.where(condition)[0])
            
        return indexs    
    
    
    def _save_indexs(self):
        indexs = []
        for j1 in range(self.p):
            index_row = []
            for j2 in range(self.p):
                index_row.append(self._save_index(j1, j2))
            indexs.append(index_row)
                
        return indexs
    
    
    def _save_all_xai_i(self):

        xais_all_i = []
        for j1 in range(self.p):
            xai_all_row_i = []
            for j2 in range(self.p):
                xai_all_i = []
                for i in range(len(self.indexs[j1][j2])):
                    xai_all_i.append(self._calc_xai_i(self.X, self.indexs[j1][j2][i][0], j1, j2))
                xai_all_row_i.append(xai_all_i)
            xais_all_i.append(xai_all_row_i)
        
        return xais_all_i
                
    
    def _rearrange_xai(self, Xai_minus_j_minus_j, Xai_minus_j_j):

        p = self.p 
        tmp_number = int((p-1) ** 2)
        Xai_minus_j_minus_j_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                Xai_minus_j_minus_j_vector[index] = Xai_minus_j_minus_j[v,q]
                index = index + 1

        for v in range(p-1):
            Xai_minus_j_minus_j_vector[tmp_number + v] = Xai_minus_j_j[v].item()
            
        return Xai_minus_j_minus_j_vector.reshape(-1,1)
    

    def generate_combinations(self, X):
        # Generate all combinations of 0 and 1 for p columns
        p = X.shape[1]
        # combinations = np.array(np.meshgrid(*[[0, 1]] * p)).T.reshape(-1, p)
        unique_rows = {tuple(row) for row in X}
        combinations = [np.array(row) for row in unique_rows]
        
        # Get the indices for all the combinations
        indices = []
        for comb in combinations:
            # Use boolean indexing to find rows in X that match the current combination
            condition = np.all(X[:, :p] == comb, axis=1)
            indices.append(np.where(condition)[0])  # Get the indices of rows that match
        
        return indices


    def _get_variance_jk(self, Sigma_minus_j_minus_j,  beta_j, j , k):
        p = self.p
        beta_j_tmp = beta_j.copy()
        # a = self.get_varaince_whole_mat()
        # log.info(f'variance_mat is {a}')
        
        if j < k:
            k = k-1
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
        else :
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
            
        tmp_number = int((p-1) ** 2)
        weighting_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                weighting_vector[index] = -(Sigma_minus_j_minus_j[k,v] * beta_j_tmp[q]).item()
                index += 1 

        for v in range(p-1):
            weighting_vector[tmp_number + v] = Sigma_minus_j_minus_j[k, v]
        

        ###
        # For all binary variables, there are only 2^p possible values
        variance = 0
        meshgrid_indices = self.generate_combinations(self.X) 
        for i in range(len(meshgrid_indices)):
            Xai_minus_j_minus_j_i = self._obtain_xai_all_minusj_minusj_i(meshgrid_indices[i][0], j)
            Xai_minus_j_j_i = self._obtain_xai_all_minusj_j_i(meshgrid_indices[i][0], j)
            Xai_vector = self._rearrange_xai(Xai_minus_j_minus_j_i, Xai_minus_j_j_i)
            tmp_scalar = weighting_vector.T @ Xai_vector
            tmp_scalar = tmp_scalar ** 2
            tmp_scalar = tmp_scalar * len(meshgrid_indices[i])
            variance = variance + tmp_scalar
        variance = variance/self.n
        
        return variance        
        
                    
    def _true_beta_from_cov(self, cov, j):
        
        cov = np.array(cov)
        XTX = np.delete(np.delete(cov, j , axis=1), j, axis=0)
        XTY = np.delete(cov[:,j], j, axis=0)
        
        return np.linalg.inv(XTX) @ XTY
    
    
    def _conditional_inference(self, j, k):
        X = self.X.copy()
        theta = self._calc_sigma_mat_minus_j_inv(j)
        beta_j = theta @ self._calc_sigma_vector_minusj_j(j)
        
        var_jk = self._get_variance_jk(theta, beta_j, j , k)
        if k > j:
            beta_jk = beta_j[k-1]
        else:
            beta_jk = beta_j[k]
            
        z_score = beta_jk*np.sqrt(self.n)/np.sqrt(var_jk)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("Variance:", var_jk)
        print("Z-score:", z_score)
        print("P-value:", p_value)
        
        return p_value    
    

    def _independence_inference(self, j1, j2):

        X = self.X
        n = self.n 
        p = self.p     

        estimator = self.estimation
        
        psi_matrix_summation = np.zeros((3,3))
        psi_vector = Psi_vector(X, j1, j2)
        for i in range(n):
            psi_matrix = psi_vector.Psi_theta_matrix(i)
            psi_matrix_summation = psi_matrix + psi_matrix_summation
        psi_matrix_average = psi_matrix_summation/n    

        # print(psi_matrix_average, '\n')
        psi_grad_matrix = self.grad_all[self.index_mapping[j1, j2]]
        psi_grad_matrix_inv = self.grad_all_inv[self.index_mapping[j1, j2]]

        variance_vector = psi_grad_matrix_inv @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
                
        # print("The estimated covaraince is ", sigma_hat)
        sigma_hat = estimator._calc_sigma_hat(j1, j2)
        z_score = (sigma_hat*np.sqrt(n) - 0)/np.sqrt(variance_vector[0,0])
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate the p-value for a two-tailed test
        # print("Variance:", variance_vector[0,0], variance_vector[1,1], variance_vector[2,2])
        # print("Z-score:", z_score)
        print("Variance:", variance_vector[0,0])
        print("P-value:", p_value)

        return p_value 
    
    
    
class Estimation_binary():
    def __init__(self, X_obs):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.h_hat_all = self._save_h_hat()
        self.pi_hat_all = self._save_pi_hat()
        self.pi_hat_j_all = self._save_pi_hat_j()
        self.sigma_hat_all = self._calc_all_sigma()
        self.grad_all, self.grad_all_inv = self._save_grad_all()
        

    def _calc_h_hat(self,j):
        X = self.X
        ratio = 1 - (X[:,j].sum()/self.n)
        h_hat = stats.norm.ppf(ratio)

        return h_hat

    def _calc_pi_hat_j(self,j):
        X = self.X
        ratio = X[:,j].sum()/self.n
        pi_hat_j = ratio

        return pi_hat_j

    def _calc_pi_hat(self,j1,j2):
        X = self.X
        ratio = (X[:,j1]*X[:,j2]).sum()/self.n
        pi_hat = ratio

        return pi_hat

    def calc_pi_hat_func(self, h_hat_j1, h_hat_j2, sigma):
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)
        mean = [0, 0]  # Means for x and y
        # Upper limits are infinity (representing the tail of the distribution)
        upper_limit = np.array([np.inf, np.inf])
        # Lower limits
        lower_limit = np.array([h_hat_j1, h_hat_j2])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        pi_hat = prob

        return pi_hat

    def _calc_sigma_hat(self, j1, j2):
        h_hat_j1 = self._calc_h_hat(j1)
        h_hat_j2 = self._calc_h_hat(j2)        
        # determine X_j1 is continous or not 
        if len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) == 2:
            pi_hat = self._calc_pi_hat(j1, j2)
            obj = lambda sigma_hat: (self.calc_pi_hat_func(h_hat_j1,h_hat_j2, sigma_hat) - pi_hat)**2
            sigma_hat = fminbound(obj, -1, 1)

        return sigma_hat

    def _calc_all_sigma(self): # return all sigma_hat as a matrix
        p = self.p
        sigma_hat_mat = np.zeros((p,p))
        for i in range(p):
            for j in range(p):
                if i == j:
                    sigma_hat_mat[i,j] = 1
                else:
                    sigma_hat_mat[i,j] = self._calc_sigma_hat(i,j)
        
        return sigma_hat_mat
    
    def _calc_sigma_hat_j(self, j):
        sigma_hat_j = np.zeros((self.p-1,1))
        j_others = [j1 for j1 in range(self.p) if j1 != j]
        for i in range(self.p-1):
            sigma_hat_j[i] = self._calc_sigma_hat(j, j_others[i])

        return np.array(sigma_hat_j).reshape(-1,1)
    
    def _save_pi_hat(self):
        p = self.p
        pi_hat = np.zeros((p,p))
        for i in range(p):
            for j in range(p):
                if i == j:
                    pi_hat[i,j] = self._calc_pi_hat_j(i)
                else:
                    pi_hat[i,j] = self._calc_pi_hat(i,j)
        
        return pi_hat
    
    def _save_pi_hat_j(self):
        p = self.p
        pi_hat = np.zeros((p,1))
        for i in range(p):
            pi_hat[i] = self._calc_pi_hat_j(i)
        
        return pi_hat
    
    def _save_h_hat(self):
        p = self.p
        h_hat = np.zeros((p,1))
        for i in range(p):
            h_hat[i] = self._calc_h_hat(i)
        
        return h_hat

    def psi_vector_i(self, i, j1, j2):
        psi_theta_1 = self.X[i,j1] * self.X[i,j2] - self.pi_hat_all[j1,j2]
        psi_theta_2 = self.X[i,j1] - self.pi_hat_j_all[j1]
        psi_theta_3 = self.X[i,j2] - self.pi_hat_j_all[j2]
        
        return np.array([psi_theta_1, psi_theta_2, psi_theta_3]).reshape(-1,1)
    
    def psi_matrix_i(self, i, j1, j2):
        psi_vector = self.psi_vector_i(i, j1, j2)
        psi_matrix = psi_vector @ psi_vector.T

        return psi_matrix
    
    def block_prob(self, h_j1, h_j2, sigma):
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)
        mean = [0, 0]
        upper_limit = np.array([np.inf, np.inf])
        lower_limit = np.array([h_j1, h_j2])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        
        return prob
    
    # calculate the gradient of psi vector using zero order gradient
    # We have three parameters: sigma, h_hat_j1, h_hat_j2
    def _calc_grad(self, j1, j2, h=1e-5):
        h_hat_j1 = self.h_hat_all[j1]
        h_hat_j2 = self.h_hat_all[j2]
        sigma_hat = self.sigma_hat_all[j1,j2]
        
        theta = np.concatenate((np.array([sigma_hat]), h_hat_j1, h_hat_j2))
        num_vars = len(theta)
        grad = np.zeros_like(theta)
        
        for k in range(num_vars):
            # Perturbation vector
            e = np.zeros(num_vars)
            e[k] = h
            
            # Perturb variables
            theta_plus = theta + e
            theta_minus = theta - e
            
            # Split variables back
            sigma_plus = theta_plus[0]
            h_hat_j1_plus = theta_plus[1:len(h_hat_j1)+1]
            h_hat_j2_plus = theta_plus[len(h_hat_j1)+1:]
            
            sigma_minus = theta_minus[0]
            h_hat_j1_minus = theta_minus[1:len(h_hat_j1)+1]
            h_hat_j2_minus = theta_minus[len(h_hat_j1)+1:]
            
            
            # Compute perturbed probabilities
            P_plus = self.block_prob(h_hat_j1_plus, h_hat_j2_plus, sigma_plus)
            P_minus = self.block_prob(h_hat_j1_minus, h_hat_j2_minus, sigma_minus)
            
            # Estimate gradient
            grad[k] = (P_plus - P_minus) / (2 * h)
            
        grad = -grad
        grad = grad.flatten().tolist()  
        
        grad_21 = 0.0 
        grad_22 = np.exp(- h_hat_j1**2 / 2)/np.sqrt(2 * np.pi)
        grad_23 = 0.0 
        grad_31 = 0.0 
        grad_32 = 0.0 
        grad_33 = np.exp(- h_hat_j2**2 / 2)/np.sqrt(2 * np.pi)

        grad_matrix = np.array([
            grad,
            [grad_21, grad_22.item(), grad_23],
            [grad_31, grad_32, grad_33.item()]
        ]).astype(np.float64)

        return grad_matrix
    
    def _save_grad_all(self):
        p = self.p 
        number_grad = int(p*p)
        grad_all = []
        grad_all_inv = []
        index = 0 
        for j1 in range(p):
            for j2 in range(p):
                grad_all.append(self._calc_grad(j1, j2))
                # print(grad_all[int(j1 * p - (j1 * (j1 + 1)) / 2 + j2)])
                grad_all_inv.append(self.inv_matrix(grad_all[index]))
                index += 1

        return grad_all, grad_all_inv        

    def inv_matrix(self, matrix):
        try:
            return np.linalg.inv(matrix)
        except np.linalg.LinAlgError:
            return np.linalg.pinv(matrix)    
     
    
class disTest():
    def __init__(self, data):
        self.X = data.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.estimation = Estimation(self.X)
        self.sigma_hat_mat = self.estimation._calc_all_sigma()
        self.grad_all, self.grad_all_inv = self._save_all_grad()
        # self.index_mapping = self.create_upper_triangle_matrix(self.p)
        self.index_mapping = np.arange(self.p*self.p).reshape(self.p, self.p)

    def get_varaince(self, X, j1, j2):
        X = X.copy()
        n = X.shape[0]
        psi_matrix_summation = np.zeros((3,3))
        psi_vector = Psi_vector(X, j1, j2)
        for i in range(n):
            psi_matrix = psi_vector.Psi_theta_matrix(i)
            psi_matrix_summation = psi_matrix + psi_matrix_summation
        psi_matrix_average = psi_matrix_summation/n    
        psi_grad = Psi_grad(X, j1, j2)
        psi_grad_matrix = psi_grad.Psi_grad_matrix()
        variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)

        return variance_vector[0,0]


    # return a p-1xp-1 matrix using parallel computing
    def get_varaince_all(self, j):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        X_tmp = np.delete(X_tmp, j, axis=1) 
        variance_all = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p-1) for j2 in range(p-1))
        variance_all = np.array(variance_all).reshape(p-1, p-1)
        
        return variance_all


    def get_varaince_whole_mat(self):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        variance_mat = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p) for j2 in range(p))
        variance_mat = np.array(variance_mat).reshape(p, p)

        return variance_mat


    # return a p-1 vector using parallel computing
    def get_varaince_vector(self, j):
        X = self.X.copy()
        p = X.shape[1]
        variance_vector = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self.get_varaince)(X, j1, j) for j1 in range(p) if j1 != j)
        variance_vector = np.array(variance_vector).reshape(p-1,1)

        return variance_vector

    def parallel_process(self, args):
        i, j = args
        # Assuming these methods can be called independently and are picklable
        Xai_minus_j_minus_j_i = self._calc_xai_all_minusj_minusj_i(i, j)
        Xai_minus_j_j_i = self._calc_xai_all_minusj_j_i(i, j)
        Xai_vector = self._rearrange_xai(Xai_minus_j_minus_j_i, Xai_minus_j_j_i)
        return Xai_vector @ Xai_vector.T
    

    def _save_all_grad(self):
        p = self.p 
        number_grad = int(p*p)
        grad_all = []
        grad_all_inv = []
        index = 0 
        for j1 in range(p):
            for j2 in range(p):
                psi_grad = choose_Psi_grad(self.X, j1, j2)
                if _trigger_stable(self.X, j1, j2):
                    print('triggered,' )
                    grad_all.append(psi_grad.Psi_grad_matrix() + 1e-4* np.eye(len(psi_grad.Psi_grad_matrix())))
                else:
                    # print(index, 'not triggered')
                    grad_all.append(psi_grad.Psi_grad_matrix())
                # print(index)
                grad_all_inv.append(np.linalg.inv(grad_all[index]))
                index += 1

        return grad_all, grad_all_inv


    def _calc_sigma_mat_minus_j(self, j): 
        # return the sigma matrix without the jth row and column
        return np.delete(np.delete(self.sigma_hat_mat, j, 0), j, 1)
    

    def _calc_sigma_mat_minus_j_inv(self,j):

        return np.linalg.inv(self._calc_sigma_mat_minus_j(j))
    
    
    def _calc_sigma_vector_minusj_j(self, j):

        return self.estimation._calc_sigma_hat_j(j)
    

    def _calc_xai_i(self, X, i, j1, j2):
        p = self.p
        psi_vector_ins = choose_Psi(X, j1, j2)
        psi_vector = psi_vector_ins.Psi_theta_vector(i)
        # if j1> j2:
        #     j1, j2 = j2, j1
        psi_grad_inv = self.grad_all_inv[self.index_mapping[j1, j2]]
        # psi_grad_inv = self.grad_all_inv[j1, j2]
        xai_i = psi_grad_inv @ psi_vector

        return xai_i[0] 
    
    
    # return the matrix of (sigma_hat_minusj_minusj - sigma_minusj_minusj)_i
    def _calc_xai_all_minusj_minusj_i(self, i, j):
        X_tmp = self.X.copy()
        # X_tmp = np.delete(X_tmp, j, axis=1)
        xai_all = Parallel(n_jobs=-1)(delayed(self._calc_xai_i)(X_tmp, i, j1, j2) for j1 in range(self.p) for j2 in range(self.p))
        xai_all_minusj = np.array(xai_all).reshape(self.p, self.p)
        xai_all_minusj = np.delete(np.delete(xai_all_minusj, j, axis=1), j, axis=0)

        return xai_all_minusj

    
    # return the vector of sigma_hat_minusj_j - sigma_minusj_j
    def _calc_xai_all_minusj_j_i(self, i, j):
        xai_all = Parallel(n_jobs=-1)(delayed(self._calc_xai_i)(self.X, i, j1, j) for j1 in range(self.p))
        xai_all = np.array(xai_all).reshape(self.p,1)
        xai_all = np.delete(xai_all, j, axis=0)

        return xai_all
    
    
    def _true_beta_from_cov(self, cov, j):
        
        cov = np.array(cov)
        XTX = np.delete(np.delete(cov, j , axis=1), j, axis=0)
        XTY = np.delete(cov[:,j], j, axis=0)
        
        return np.linalg.inv(XTX) @ XTY
    
    
    def _rearrange_xai(self, Xai_minus_j_minus_j, Xai_minus_j_j):

        p = self.p 
        tmp_number = int((p-1) ** 2)
        Xai_minus_j_minus_j_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                Xai_minus_j_minus_j_vector[index] = Xai_minus_j_minus_j[v,q]
                index = index + 1

        for v in range(p-1):
            Xai_minus_j_minus_j_vector[tmp_number + v] = Xai_minus_j_j[v].item()
            
        return Xai_minus_j_minus_j_vector.reshape(-1,1)
    

    def _get_variance_jk(self, Sigma_minus_j_minus_j,  beta_j, j , k):
        # Rearrange the matrix to be shape with px(p+1)/2 - 1 vector as Xai_minus_j_minus_j is symmetric
        p = self.p
        beta_j_tmp = beta_j.copy()
        
        if j < k:
            k = k-1
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
        else :
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
            
        tmp_number = int((p-1) ** 2)
        weighting_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                weighting_vector[index] = -(Sigma_minus_j_minus_j[k,v] * beta_j_tmp[q]).item()
                index += 1 

        for v in range(p-1):
            weighting_vector[tmp_number + v] = Sigma_minus_j_minus_j[k, v]
        
        # print(weighting_vector)
        # rearrange the Xai_minus_j_minus_j to be a vector of length px(p+1)/2 - 1
        Xai_mat_summation = np.zeros([int((p-1)** 2 + p-1), int((p-1)** 2 + p-1)])
        
        with Pool() as pool:
            results = pool.map(self.parallel_process, [(i, j) for i in range(self.n)])
            for result in results:
                Xai_mat_summation += result
        Xai_mat_avg = Xai_mat_summation/self.n    
        # print(Xai_mat_avg)
        variance = weighting_vector.T @ Xai_mat_avg @ weighting_vector

        return variance
    

    def _conditional_inference(self, j, k):
        X = self.X.copy()
        theta = self._calc_sigma_mat_minus_j_inv(j)
        beta_j = theta @ self._calc_sigma_vector_minusj_j(j)       
        
        var_jk = self._get_variance_jk(theta, beta_j, j , k)
        if k > j:
            beta_jk = beta_j[k-1]
        else:
            beta_jk = beta_j[k]
            
        z_score = beta_jk*np.sqrt(self.n)/np.sqrt(var_jk)
        # z_score = beta_jk * self.n/np.sqrt(var_jk)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("Variance:", var_jk)
        # print("Z-score:", z_score)
        print("P-value:", p_value)
        
        return p_value
    

    def _independence_inference(self, j1, j2):

        X = self.X
        n = self.n 
        p = self.p     

        estimator = self.estimation
        
        if len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) == 2:
            psi_matrix_summation = np.zeros((3,3))
            psi_vector = Psi_vector(X, j1, j2)
            for i in range(n):
                psi_matrix = psi_vector.Psi_theta_matrix(i)
                psi_matrix_summation = psi_matrix + psi_matrix_summation
            psi_matrix_average = psi_matrix_summation/n    

            # print(psi_matrix_average, '\n')
            psi_grad = Psi_grad(X, j1, j2)
            psi_grad_matrix = psi_grad.Psi_grad_matrix()
            variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
                
            
        elif len(set(self.X[:,j1])) > 2 and len(set(self.X[:,j2])) == 2:
            psi_matrix_summation = np.zeros((2,2))
            psi_vector = Psi_vector_mixed(X, j1, j2)
            for i in range(n):
                psi_matrix = psi_vector.Psi_theta_matrix(i)
                psi_matrix_summation = psi_matrix + psi_matrix_summation
            psi_matrix_average = psi_matrix_summation/n    

            # print(psi_matrix_average, '\n')
            psi_grad = Psi_grad_mixed(X, j1, j2)
            psi_grad_matrix = psi_grad.Psi_grad_matrix()
            # print(psi_grad_matrix, '\n')
            variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
            
            
        elif len(set(self.X[:,j1])) == 2 and len(set(self.X[:,j2])) > 2:
            psi_matrix_summation = np.zeros((2,2))
            psi_vector = Psi_vector_mixed(X, j2, j1)
            for i in range(n):
                psi_matrix = psi_vector.Psi_theta_matrix(i)
                psi_matrix_summation = psi_matrix + psi_matrix_summation
            psi_matrix_average = psi_matrix_summation/n
            
            psi_grad = Psi_grad_mixed(X, j2, j1)
            psi_grad_matrix = psi_grad.Psi_grad_matrix()
            psi_grad_matrix_inv = np.linalg.inv(psi_grad_matrix)
            variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
        
        else:
            variance_vector = Psi_con(X, j1, j2).Var()
            variance_vector = variance_vector.reshape(1,1)
            
        sigma_hat = estimator._calc_sigma_hat(j1, j2)
        z_score = (sigma_hat*np.sqrt(n) - 0)/np.sqrt(variance_vector[0,0])
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("Variance:", variance_vector[0,0])
        print("P-value:", p_value)

        return p_value            
    

class disTest_binary():
    def __init__(self, X_obs):

        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.index_mapping = np.arange(self.p*self.p).reshape(self.p, self.p)
    
        self.estimation = Estimation_binary(self.X)
        self.sigma_hat_mat = self.estimation.sigma_hat_all
        self.grad_all, self.grad_all_inv = self.estimation.grad_all, self.estimation.grad_all_inv 
        self.indexs = self._save_indexs()   # indexs of zero and one for each j1, j2
        self.look_up_tables = self.look_up_table_all()
        self.xai_all = self._save_all_xai_i()
        
   
    def look_up_table(self, j1, j2):
        look_up_dict = {}
        for idx, i in enumerate(self.indexs[j1][j2]):
            for number in i:
                look_up_dict[number] = idx
                
        return look_up_dict
    
    
    def look_up_table_all(self):
        look_up_table_all = []
        for j1 in range(self.p):
            look_up_table_row = []
            for j2 in range(self.p):
                look_up_table_row.append(self.look_up_table(j1, j2))
            look_up_table_all.append(look_up_table_row) 
        
        return look_up_table_all
    
        
    def get_varaince(self, X, j1, j2):
        X = X.copy()
        n = X.shape[0]
        psi_vector = choose_Psi(X, j1, j2)
        psi_matrix_summation = np.zeros_like(psi_vector.Psi_theta_matrix(0))
        for i in range(n):
            psi_matrix = psi_vector.Psi_theta_matrix(i)
            psi_matrix_summation = psi_matrix + psi_matrix_summation
        psi_matrix_average = psi_matrix_summation/n    
        psi_grad = choose_Psi_grad(X, j1, j2)
        psi_grad_matrix = psi_grad.Psi_grad_matrix()
        variance_vector = np.linalg.inv(psi_grad_matrix) @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)

        return variance_vector[0,0]


    # return a p-1xp-1 matrix using parallel computing
    def get_varaince_all(self, j):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        X_tmp = np.delete(X_tmp, j, axis=1) 
        variance_all = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p-1) for j2 in range(p-1))
        variance_all = np.array(variance_all).reshape(p-1, p-1)
        
        return variance_all


    def get_varaince_whole_mat(self):
        X_tmp = self.X.copy()
        p = X_tmp.shape[1]
        variance_mat = Parallel(n_jobs=-1, backend ='multiprocessing')(delayed(self.get_varaince)(X_tmp, j1, j2) for j1 in range(p) for j2 in range(p))
        variance_mat = np.array(variance_mat).reshape(p, p)

        return variance_mat


    # return a p-1 vector using parallel computing
    def get_varaince_vector(self, j):
        X = self.X.copy()
        p = X.shape[1]
        variance_vector = Parallel(n_jobs=-1, backend='multiprocessing')(delayed(self.get_varaince)(X, j1, j) for j1 in range(p) if j1 != j)
        variance_vector = np.array(variance_vector).reshape(p-1,1)

        return variance_vector


    def _calc_sigma_mat_minus_j(self, j): 
        # return the sigma matrix without the jth row and column
        return np.delete(np.delete(self.sigma_hat_mat, j, 0), j, 1)
    

    def _calc_sigma_mat_minus_j_inv(self,j):

        return np.linalg.inv(self._calc_sigma_mat_minus_j(j))
    
    
    def _calc_sigma_vector_minusj_j(self, j):

        return self.estimation._calc_sigma_hat_j(j)
    

    def _calc_xai_i(self, X, i, j1, j2):
        p = self.p
        psi_vector_ins = choose_Psi(X, j1, j2)
        psi_vector = psi_vector_ins.Psi_theta_vector(i)
        # if j1> j2:
        #     j1, j2 = j2, j1
        psi_grad_inv = self.grad_all_inv[self.index_mapping[j1, j2]]
        # psi_grad_inv = self.grad_all_inv[j1, j2]
        xai_i = psi_grad_inv @ psi_vector

        return xai_i[0] 
    

    def _obtain_xai_i(self, i, j1, j2):
        i_belongs_to = self.look_up_tables[j1][j2][i]
        xai_i = self.xai_all[j1][j2][i_belongs_to]
        
        return xai_i
    
    
    def _obtain_xai_all_minusj_minusj_i(self, i, j):
        xai_all = np.zeros((self.p, self.p))
        for j1 in range(self.p):
            for j2 in range(self.p):
                xai_all[j1, j2] = self._obtain_xai_i(i, j1, j2)
                
        xai_all_minusj = np.delete(np.delete(xai_all, j, axis=1), j, axis=0)
        
        return xai_all_minusj
    
    
    def _obtain_xai_all_minusj_j_i(self, i, j):
        xai_all = np.zeros((self.p, 1))
        for j1 in range(self.p):
            xai_all[j1] = self._obtain_xai_i(i, j1, j)
        xai_all = np.delete(xai_all, j, axis=0)
        
        return xai_all    
    

    def _save_index(self, j1, j2):
        indexs = []
        unique_rows = {tuple(row) for row in self.X[:,[j1, j2]]}
        combinations = [np.array(row) for row in unique_rows]
        for i in range(len(combinations)):
            condition = np.all(self.X[:,[j1, j2]] == combinations[i], axis=1)
            indexs.append(np.where(condition)[0])
            
        return indexs    
    
    
    def _save_indexs(self):
        indexs = []
        for j1 in range(self.p):
            index_row = []
            for j2 in range(self.p):
                index_row.append(self._save_index(j1, j2))
            indexs.append(index_row)
                
        return indexs
    
    
    def _save_all_xai_i(self):

        xais_all_i = []
        for j1 in range(self.p):
            xai_all_row_i = []
            for j2 in range(self.p):
                xai_all_i = []
                for i in range(len(self.indexs[j1][j2])):
                    xai_all_i.append(self._calc_xai_i(self.X, self.indexs[j1][j2][i][0], j1, j2))
                xai_all_row_i.append(xai_all_i)
            xais_all_i.append(xai_all_row_i)
        
        return xais_all_i
                
    
    def _rearrange_xai(self, Xai_minus_j_minus_j, Xai_minus_j_j):

        p = self.p 
        tmp_number = int((p-1) ** 2)
        Xai_minus_j_minus_j_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                Xai_minus_j_minus_j_vector[index] = Xai_minus_j_minus_j[v,q]
                index = index + 1

        for v in range(p-1):
            Xai_minus_j_minus_j_vector[tmp_number + v] = Xai_minus_j_j[v].item()
            
        return Xai_minus_j_minus_j_vector.reshape(-1,1)
    

    def generate_combinations(self, X):
        # Generate all combinations of 0 and 1 for p columns
        p = X.shape[1]
        # combinations = np.array(np.meshgrid(*[[0, 1]] * p)).T.reshape(-1, p)
        unique_rows = {tuple(row) for row in X}
        combinations = [np.array(row) for row in unique_rows]
        
        # Get the indices for all the combinations
        indices = []
        for comb in combinations:
            # Use boolean indexing to find rows in X that match the current combination
            condition = np.all(X[:, :p] == comb, axis=1)
            indices.append(np.where(condition)[0])  # Get the indices of rows that match
        
        return indices


    def _get_variance_jk(self, Sigma_minus_j_minus_j,  beta_j, j , k):
        p = self.p
        beta_j_tmp = beta_j.copy()
        # a = self.get_varaince_whole_mat()
        # log.info(f'variance_mat is {a}')
        
        if j < k:
            k = k-1
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
        else :
            beta_j_tmp[k] = 0
            # print('the term of beta_j is', beta_j_tmp)  
            
        tmp_number = int((p-1) ** 2)
        weighting_vector = np.zeros(int((p-1)** 2 + p-1))
        index = 0
        for v in range(p-1):
            for q in range(p-1):
                weighting_vector[index] = (Sigma_minus_j_minus_j[k,v] * beta_j_tmp[q]).item()
                index += 1 

        for v in range(p-1):
            weighting_vector[tmp_number + v] = -Sigma_minus_j_minus_j[k, v]
        

        ###
        # For all binary variables, there are only 2^p possible values
        variance = 0
        meshgrid_indices = self.generate_combinations(self.X) 
        for i in range(len(meshgrid_indices)):
            Xai_minus_j_minus_j_i = self._obtain_xai_all_minusj_minusj_i(meshgrid_indices[i][0], j)
            Xai_minus_j_j_i = self._obtain_xai_all_minusj_j_i(meshgrid_indices[i][0], j)
            Xai_vector = self._rearrange_xai(Xai_minus_j_minus_j_i, Xai_minus_j_j_i)
            tmp_scalar = weighting_vector.T @ Xai_vector
            tmp_scalar = tmp_scalar ** 2
            tmp_scalar = tmp_scalar * len(meshgrid_indices[i])
            variance = variance + tmp_scalar
        variance = variance/self.n
        
        return variance        
        
                    
    def _true_beta_from_cov(self, cov, j):
        
        cov = np.array(cov)
        XTX = np.delete(np.delete(cov, j , axis=1), j, axis=0)
        XTY = np.delete(cov[:,j], j, axis=0)
        
        return np.linalg.inv(XTX) @ XTY
    
    
    def _conditional_inference(self, j, k):
        X = self.X.copy()
        theta = self._calc_sigma_mat_minus_j_inv(j)
        beta_j = theta @ self._calc_sigma_vector_minusj_j(j)
        # beta_j_true = self._true_beta_from_cov(self.cov, j)        
        
        var_jk = self._get_variance_jk(theta, beta_j, j , k)
        if k > j:
            beta_jk = beta_j[k-1]
        else:
            beta_jk = beta_j[k]
            
        z_score = beta_jk*np.sqrt(self.n)/np.sqrt(var_jk)
        # z_score = beta_jk * self.n/np.sqrt(var_jk)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("Variance:", var_jk)
        print("Z-score:", z_score)
        print("P-value:", p_value)
        
        return p_value    
    

    def _independence_inference(self, j1, j2):

        X = self.X
        n = self.n 
        p = self.p     

        estimator = self.estimation
        
        psi_matrix_summation = np.zeros((3,3))
        psi_vector = Psi_vector(X, j1, j2)
        for i in range(n):
            psi_matrix = psi_vector.Psi_theta_matrix(i)
            psi_matrix_summation = psi_matrix + psi_matrix_summation
        psi_matrix_average = psi_matrix_summation/n    

        # print(psi_matrix_average, '\n')
        psi_grad_matrix = self.grad_all[self.index_mapping[j1, j2]]
        psi_grad_matrix_inv = self.grad_all_inv[self.index_mapping[j1, j2]]

        variance_vector = psi_grad_matrix_inv @ psi_matrix_average @ np.linalg.inv(psi_grad_matrix.T)
                
        # print("The estimated covaraince is ", sigma_hat)
        sigma_hat = estimator._calc_sigma_hat(j1, j2)
        z_score = (sigma_hat*np.sqrt(n) - 0)/np.sqrt(variance_vector[0,0])
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        # Calculate the p-value for a two-tailed test
        # print("Variance:", variance_vector[0,0], variance_vector[1,1], variance_vector[2,2])
        # print("Z-score:", z_score)
        print("Variance:", variance_vector[0,0])
        print("P-value:", p_value)

        return p_value 
    
    

class DCT:
    def __new__(cls, X):
        # Determine the data type and output using a helper function
        data_type, X_output = cls.determine_type(X)
        
        # Call distest_b or distest based on the data type
        if data_type == 'binary':
            return disTest_binary(X_output)
        else:
            return disTest(X_output)

    @staticmethod
    def determine_type(X):
        cardinalities = []
        for col_idx in range(X.shape[1]):
            unique_vals = np.unique(X[:, col_idx])
            cardinalities.append(len(unique_vals))
        
        # Find all dimension indices with cardinality <= 5
        all_le_5 = all(c <= 5 for c in cardinalities)
        all_gt_5 = all(c > 5 for c in cardinalities)

        if all_le_5:
            data_type = "binary"
            # Binarize every column
            X_output = _binary_based_mean(X, np.arange(X.shape[1]))
        elif all_gt_5:
            data_type = "continuous"
            print("The data is continuous !!! You should not use DCT at all !!!")
            # No binarization - return a copy of the original
            X_output = X.copy()
        else:
            dims_le_5 = [idx for idx, c in enumerate(cardinalities) if c <= 5]        
            data_type = "mixed"
            # Binarize only columns with cardinality <= 5
            X_output = _binary_based_mean(X, dims_le_5)

        return data_type, X_output
            