import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from scipy.stats import kendalltau
from scipy import stats
from scipy.optimize import fminbound
from scipy.stats import mvn
import sympy as sp
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor
import time
import logging as log
from scipy.stats import mvn
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
from joblib import Parallel, delayed
from scipy import stats
import sympy as sp
import multiprocessing
from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor




class DCT_GMM():
    def __init__(self, X_obs, phase_two = False):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.est = Estimation(self.X)
        self.gmm = TwoStepGMM(self.X)
        self.phase_two = phase_two
        if phase_two:
            self.Sigma_phase_two = self._calc_Sigma_phase_two()
            self.GTAGinvGTA, self.G = self._save_all_GTAGinvGTA_phase_two()
            self.A_phase_two = [[None for _ in range(self.p)] for _ in range(self.p)]
            # np.fill_diagonal(self.Sigma_phase_two, 1.0)
        else:
            self.Sigma_phase_one = self.gmm.Sigma_phase_one
            # np.fill_diagonal(self.Sigma_phase_one, 1.0)
            # print("Self Sigma phase one is", self.Sigma_phase_one)
            self.GTAGinvGTA, self.G = self._save_all_GTAGinvGTA()
        self.indexs = self._save_indexs()
        self.xai_all = self._save_all_xai_i()    
        self.look_up_tables = self.look_up_table_all()       


    # Indices from block_indx_matrix, and the other one who is all zero
    def _save_indexs(self):
        indexs = []
        block_indx_matrix = self.est.block_indx_matrix
        n = self.n
        for j1 in range(self.p):
            index_row = []
            for j2 in range(self.p):
                index = []
                for l, k in block_indx_matrix[j1][j2]:
                    index.append(np.where((self.X[:, j1] == l) & (self.X[:, j2] == k))[0])
                included_indices = set(np.concatenate(index))
                all_indices = set(range(n))
                excluded_indices = all_indices - included_indices
                index.append(np.array(list(excluded_indices)))
                index_row.append(index)
            indexs.append(index_row)
        
        return indexs


    def look_up_table_all(self):
        look_up_table = []
        for j1 in range(self.p):
            look_up_table_row = []
            for j2 in range(self.p):
                look_up_table_row.append(self.look_up_table(j1, j2))
            look_up_table.append(look_up_table_row)
        return look_up_table
        

    def look_up_table(self, j1, j2):
        look_up_dict = {}
        for idx, i in enumerate(self.indexs[j1][j2]):
            for number in i:
                look_up_dict[number] = idx
                
        return look_up_dict


    def _calc_Sigma_phase_two(self):
        Sigma = np.zeros((self.p, self.p))
        for j1 in range(self.p):
            for j2 in range(self.p):
                Omega_hat = self.est._calc_Omega_hat(j1, j2)
                A = np.linalg.inv(Omega_hat)
                Sigma[j1, j2] = self.est._calc_sigma_phase_two(j1,j2, A)
        return Sigma
        

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
    

    def _invert_matrix(self, matrix):
        if matrix.shape[0] == matrix.shape[1] and np.linalg.det(matrix) != 0:
            # Matrix is invertible
            return np.linalg.inv(matrix)
        else:
            # Matrix is not invertible, use the pseudo-inverse
            return np.linalg.pinv(matrix)        


    def _save_all_GTAGinvGTA(self):
        p = self.p
        GTAGinvGTA = []
        G = []
        for j1 in range(p):
            new_row_GTAGinvGTA = []
            new_row_G = []
            for j2 in range(p):
                tmp = self.gmm._calc_grad(j1, j2, self.Sigma_phase_one[j1, j2])
                if j1 != j2:
                    new_row_G.append(tmp)
                    new_row_GTAGinvGTA.append(self._invert_matrix(tmp.T @ tmp) @ tmp.T)
                else: 
                    new_row_G.append(tmp)
                    new_row_GTAGinvGTA.append(self._invert_matrix(tmp.T @ tmp) @ tmp.T)
            GTAGinvGTA.append(new_row_GTAGinvGTA)
            G.append(new_row_G) 

        return GTAGinvGTA, G
    

    def _save_all_GTAGinvGTA_phase_two(self):
        p = self.p 
        GTAGinvGTA = []
        G = []        
        for j1 in range(p):
            new_row_GTAGinvGTA = []
            new_row_G = []
            for j2 in range(p):
                Omega_hat_phase_two = self.est._calc_Omega_hat_phase_two(j1, j2, self.Sigma_phase_two[j1, j2])
                A_phase_two = np.linalg.inv(Omega_hat_phase_two)
                tmp = self.gmm._calc_grad(j1, j2, self.Sigma_phase_two[j1, j2])
                # print(tmp.T @ A_phase_two @ tmp)
                if j1 != j2:
                    new_row_G.append(tmp)
                    new_row_GTAGinvGTA.append(self._invert_matrix(tmp.T @ A_phase_two @ tmp) @ tmp.T @ A_phase_two)
                else: 
                    new_row_G.append(tmp)
                    new_row_GTAGinvGTA.append(self._invert_matrix(tmp.T @ A_phase_two @ tmp) @ tmp.T @ A_phase_two)
            GTAGinvGTA.append(new_row_GTAGinvGTA)
            G.append(new_row_G) 
        
        return GTAGinvGTA, G        


    def _calc_sigma_mat_minus_j(self, j): 
        # return the sigma matrix without the jth row and column
        return np.delete(np.delete(self.Sigma_phase_one, j, 0), j, 1)
    
    def _calc_sigma_mat_minus_j_phase_two(self, j): 
        # return the sigma matrix without the jth row and column
        return np.delete(np.delete(self.Sigma_phase_two, j, 0), j, 1)
    

    def _calc_sigma_mat_minus_j_inv(self,j):

        return np.linalg.inv(self._calc_sigma_mat_minus_j(j))    
    
    def _calc_sigma_mat_minus_j_inv_phase_two(self,j):

        return np.linalg.inv(self._calc_sigma_mat_minus_j_phase_two(j))
    

    def _save_all_xai_i(self):
        indexs = self.indexs
        phase_two = self.phase_two
        xais_all_i = []
        for j1 in range(self.p):
            xai_all_row_i  = []
            for j2 in range(self.p):
                xai_all_i = []
                for i in range(len(indexs[j1][j2])):
                    if phase_two:
                        xai_i = self._calc_xai_i_phase_two(indexs[j1][j2][i][0], j1, j2)
                    else:
                        xai_i = self._calc_xai_i(indexs[j1][j2][i][0], j1, j2)
                    xai_all_i.append(xai_i) 
                
                xai_all_row_i.append(xai_all_i)
            xais_all_i.append(xai_all_row_i)
            
        return xais_all_i
            

    def _calc_xai_i(self, i, j1, j2):
        moment_condition_i = self.est._moment_condition_i(j1, j2, i)     
        # print('moment_condition_i shape is ', moment_condition_i.shape)
        GTATinvGTA = self.GTAGinvGTA[j1][j2]
        xai_i = GTATinvGTA @ moment_condition_i
        # print('xai_i shape is ', xai_i.shape)
        output = xai_i[0]

        return output
    

    def _calc_xai_i_phase_two(self, i, j1, j2):
        moment_condition_i = self.est._moment_condition_i_phase_two(j1, j2, i, self.Sigma_phase_two[j1, j2])
        GTATinvGTA = self.GTAGinvGTA[j1][j2]
        xai_i = GTATinvGTA @ moment_condition_i
        output = xai_i[0]

        return output
    
    
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


    def _calc_sigma_vector_minusj_j(self, j):
        col = self.Sigma_phase_one[:,j]
        col_without_jth_row = np.delete(col, j, axis=0)
        return col_without_jth_row        
        

    def _calc_sigma_vector_minusj_j_phase_two(self, j):
        col = self.Sigma_phase_two[:,j]
        col_without_jth_row = np.delete(col, j, axis=0)
        return col_without_jth_row         


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
        
        
        # For catrgorical cases, there are only A x B X C x ... combinations
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
        
    
    def _get_variance_jk_phase_two(self, Sigma_minus_j_minus_j,  beta_j, j , k):
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
        # For catrgorical cases, there are only A x B X C x ... combinations
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
    

    def _inference_phase_one(self, j, k):
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
        # print("Z-score:", z_score)
        print("P-value:", p_value)
        
        return p_value        
    

    def _inference_phase_two(self, j, k):
        X = self.X.copy()
        theta = self._calc_sigma_mat_minus_j_inv_phase_two(j)
        beta_j = theta @ self._calc_sigma_vector_minusj_j_phase_two(j)
        # beta_j_true = self._true_beta_from_cov(self.cov, j)        
        
        var_jk = self._get_variance_jk_phase_two(theta, beta_j, j , k)
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
    
    
    def _inference(self, j, k):
        if self.phase_two:
            return self._inference_phase_two(j, k)
        else:
            return self._inference_phase_one(j, k)
    


class Estimation():
    def __init__(self, X_obs):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.card = self._check_all_card()
        # self.tau_matrix = [[None for _ in range(self.p)] for _ in range(self.p)]
        self.boundaries_matrix, self.block_indx_matrix = self._calc_boundaries_block_matrix()
        self.tau_matrix = self._calc_tau_func_matrix()
        self._check_invertible_all_blocks()
        

    # check the cardniality of the j-th ordinal variable
    def _check_card(self, j):
        X_j = self.X[:,j]
        card = len(set(X_j))

        return card

    # check all cardinality of the ordinal variables
    def _check_all_card(self):
        card_ls = []
        for j in range(self.p):
            card = self._check_card(j)
            card_ls.append(card)
        return card_ls


    # return the vector of h_hat for j-th variables, h_hat is the boundary 
    def _calc_h_hat(self,j):
        X = self.X
        card_j = self.card[j]
        ratios = np.zeros(card_j-1)
        # return the proportion of X_j == i, i = 1,2,...,card_j
        accumulated_ratio = 0
        for i in range(card_j-1):
            ratio = (X[:,j] == i).sum()/self.n + accumulated_ratio
            if ratio == 0:
                h_hat = i
                break
            accumulated_ratio = ratio
            ratios[i] = accumulated_ratio
        h_hat = stats.norm.ppf(ratios)            
        return h_hat

    # return the vector of pi_hat for j-th variables, pi_hat is the probability of X_j == i
    def _calc_tau_hat_j(self,j):
        X = self.X
        card_j = self.card[j]
        ratios = np.zeros(card_j)
        for i in range(card_j):
            ratio = (X[:,j] == i).sum()/self.n 
            if ratio == 0:
                h_hat = i
                break
            ratios[i] = ratio        

        return ratios
    
    # averaged tau_jk 
    def _calc_tau_all(self, j1, j2):
        X = self.X
        card_j1 = self.card[j1]
        card_j2 = self.card[j2]

        tau_jk = np.zeros((card_j1, card_j2))
        for j in range(card_j1):
            for k in range(card_j2):
                # print(((X[:,j1] == j) & (X[:,j2] == k)).sum()/len(X))
                tau_jk[j,k] = ((X[:,j1] == j) & (X[:,j2] == k)).sum()/len(X)

        return tau_jk
    

    def _calc_tau_chosen_blocks(self, j1, j2, indx_ls):

        X = self.X
        card_j1 = self.card[j1]
        card_j2 = self.card[j2]
        tau_jk = np.zeros((card_j1, card_j2))      
        for j, k in indx_ls:
            tau_jk[j,k] = ((X[:,j1] == j) & (X[:,j2] == k)).sum()/len(X)
 
        tau_jk = tau_jk[tau_jk != 0].reshape(-1, 1)
        # self.tau_matrix[j1][j2] = tau_jk

        return tau_jk
    
    
    def _calc_tau_all_i(self, j1, j2, i):
        X = self.X
        card_j1 = self.card[j1]
        card_j2 = self.card[j2]

        tau_jk = np.zeros((card_j1, card_j2))
        for j in range(card_j1):
            for k in range(card_j2):
                tau_jk[j,k] = int((X[i,j1] == j) & (X[i,j2] == k))

        return tau_jk
    

    ### !!!! Important: We should not have all zero terms  
    def _calc_tau_chosen_blocks_i(self, j1, j2, i, indx_ls):
        X = self.X
        card_j1 = self.card[j1]
        card_j2 = self.card[j2]

        tau_jk = np.zeros((card_j1, card_j2))
        for j, k in indx_ls:
            tau_jk[j,k] = int((X[i,j1] == j) & (X[i,j2] == k))

        tau_jk = np.array([tau_jk[i,j] for i,j in indx_ls])
        # tau_jk = tau_jk[tau_jk != 0]
        
        return tau_jk
    
    
    # def _calc_tau_chosen_blocks_i(self, j1, j2, i):
    #     X = self.X
    #     card_j1 = self._check_card(j1)
    #     card_j2 = self._check_card(j2)

    #     tau_jk = np.zeros((card_j1, card_j2))
    #     _, indx_ls = self._choose_blocks(j1, j2)
    #     for j, k in indx_ls:
    #         tau_jk[j,k] = int((X[i,j1] == j) & (X[i,j2] == k))

    #     return tau_jk
    

    def _calc_tau_func(self, h_j1_lower_limit, h_j1_upper_limit, h_j2_lower_limit, h_j2_upper_limit, sigma):
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2) 
        mean = [0, 0]  # Means for x and y
        upper_limit = np.array([h_j1_upper_limit, h_j2_upper_limit])
        # Lower limits
        lower_limit = np.array([h_j1_lower_limit, h_j2_lower_limit])
        prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
        pi_hat = prob

        return pi_hat
    

    def _calc_tau_func_all(self, j1, j2, sigma):
        h_hat_j1 = self._calc_h_hat(j1)
        h_hat_j2 = self._calc_h_hat(j2)
        boundaries_j1 = np.concatenate([[-np.inf], h_hat_j1, [np.inf]])
        boundaries_j2 = np.concatenate([[-np.inf], h_hat_j2, [np.inf]])
        probs = np.zeros((len(boundaries_j1)-1, len(boundaries_j2)-1))
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)  
        mean = [0, 0]  # Means for x and y
        for i in range(len(boundaries_j1)-1):
            for j in range(len(boundaries_j2)-1):
                upper_limit = np.array([boundaries_j1[i+1], boundaries_j2[j+1]])
                lower_limit = np.array([boundaries_j1[i], boundaries_j2[j]])
                # print(lower_limit, upper_limit)
                # print(j,i)
                prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)

                probs[i,j] = prob
        probs = probs.reshape(-1, 1)
        # print(probs.shape)

        return probs
    

    def _calc_tau_func_chosen_blocks(self, j1, j2, sigma, boundaries_ls):
        probs = np.zeros(len(boundaries_ls))
        mat = np.array([[1, sigma], [sigma, 1]], dtype=np.double, ndmin=2)  
        mean = [0, 0]  # eans for x and y
        for i in range(len(boundaries_ls)):
            lower_limit = np.array([boundaries_ls[i][0], boundaries_ls[i][2]])
            upper_limit = np.array([boundaries_ls[i][1], boundaries_ls[i][3]])
            prob, _ = mvn.mvnun(lower_limit, upper_limit, mean, mat)
            probs[i] = prob
        probs = probs.reshape(-1, 1)

        return probs
    
    # Calculate the Sigma using A as an identity matrix
    def _calc_sigma(self, j1, j2):
        tau_j1j2 = self._calc_tau_all(j1, j2).reshape(-1, 1)
        # print(tau_j1j2.shape)
        # print(tau_j1j2)
        obj = lambda sigma: np.sum((tau_j1j2 - self._calc_tau_func_all(j1, j2, sigma))**2)
        sigma = fminbound(obj, -1, 1)

        return sigma
    
    
    # Calculate the Sigma using A as a matrix with choosen blocks
    def _calc_sigma_phase_one_chosen_blocks(self, j1, j2):
        boundaries_ls, block_indx = self._choose_blocks(j1, j2)
        tau_j1j2 = self._calc_tau_chosen_blocks(j1, j2, block_indx).reshape(-1, 1)
        # print(tau_j1j2.shape)
        # print(len(boundaries_ls))
        obj = lambda sigma: np.sum((tau_j1j2 - self._calc_tau_func_chosen_blocks(j1, j2, sigma, boundaries_ls))**2)
        sigma = fminbound(obj, -1, 1)

        return sigma

    
    def _calc_sigma_phase_two(self, j1, j2 , A):
        boundaries_ls_mat, block_indx_mat = self.boundaries_matrix, self.block_indx_matrix
        boundaries_ls = boundaries_ls_mat[j1][j2]
        block_indx = block_indx_mat[j1][j2]
        tau_j1j2 = self._calc_tau_chosen_blocks(j1, j2, block_indx).reshape(-1, 1)
        # print(tau_j1j2.shape)
        # print(A.shape)
        obj = lambda sigma: np.sum((tau_j1j2 - self._calc_tau_func_chosen_blocks(j1, j2, sigma, boundaries_ls)).T @ A @ (tau_j1j2 - self._calc_tau_func_chosen_blocks(j1, j2, sigma, boundaries_ls)))
        sigma = fminbound(obj, -1, 1)

        return sigma 

    # Due to some blocks have very small probability mass, and thus, super large variance. We need to choose the blocks that have enough probability mass
    def _choose_blocks(self, j1, j2):
        card_j1 = self.card[j1]
        card_j2 = self.card[j2]
        tau_j1j2 = self._calc_tau_all(j1, j2)
        h_hat_j1 = self._calc_h_hat(j1)
        h_hat_j2 = self._calc_h_hat(j2)
        boundaries_j1 = np.concatenate([[-np.inf], h_hat_j1, [np.inf]])
        boundaries_j2 = np.concatenate([[-np.inf], h_hat_j2, [np.inf]])
        boundary_ls = []
        indx_ls = []
        # print("We are checking the block for j1 = ", j1, "j2 = ", j2)
        for j1 in range(card_j1):          
            for j2 in range(card_j2):
                if tau_j1j2[j1,j2] < 0.01:
                    continue
                else:
                    boundary_ls.append([boundaries_j1[j1], boundaries_j1[j1+1], boundaries_j2[j2], boundaries_j2[j2+1]])
                    indx = [j1, j2]
                    indx_ls.append(indx)
        # print('The number of blocks (functions) is ', len(boundary_ls))
        return boundary_ls, indx_ls 


    def _calc_boundaries_block_matrix(self):
        boundaries_matrix = []
        block_indx_matrix = []
        for j1 in range(self.p):
            new_row_boundaries = []
            new_row_indx = []
            for j2 in range(self.p):
                boundaries_ls, block_indx = self._choose_blocks(j1, j2)
                new_row_boundaries.append(boundaries_ls)
                new_row_indx.append(block_indx)
            boundaries_matrix.append(new_row_boundaries)
            block_indx_matrix.append(new_row_indx)

        return boundaries_matrix, block_indx_matrix 
    

    def _calc_tau_func_matrix(self):
        tau_matrix = []
        for j1 in range(self.p):
            new_row = []
            for j2 in range(self.p):
                # self._calc_tau_chosen_blocks(j1, j2, self.block_indx_matrix[j1][j2])
                new_row.append(self._calc_tau_chosen_blocks(j1, j2, self.block_indx_matrix[j1][j2]))
            tau_matrix.append(new_row)
            
        return tau_matrix    
    

    def _check_invertible_all_blocks(self):
        for j1 in range(self.p):
            for j2 in range(self.p):
                self._recheck_omega_hat_invertibility(j1, j2)


    def _moment_condition_i(self, j1, j2, i):
        # print(self.block_indx_matrix[j1][j2])
        tau_jk_i = self._calc_tau_chosen_blocks_i(j1, j2, i, self.block_indx_matrix[j1][j2]).reshape(-1, 1)
        tau_jk_i_minus_tau_func = tau_jk_i - self.tau_matrix[j1][j2].reshape(-1, 1)
        # print('tau_jk shape is ', tau_jk_i.shape)
        # print('tau matrix shape is', self.tau_matrix[j1][j2].shape)

        return tau_jk_i_minus_tau_func  


    def _moment_condition_i_phase_two(self, j1, j2, i, sigma):

        tau_jk_i = self._calc_tau_chosen_blocks_i(j1, j2, i, self.block_indx_matrix[j1][j2]).reshape(-1, 1)
        tau_jk_i_minus_tau_func = tau_jk_i - self._calc_tau_func_chosen_blocks(j1, j2, sigma, self.boundaries_matrix[j1][j2])

        return tau_jk_i_minus_tau_func
    

    def _calc_Omega_hat(self, j1, j2):
        tmp = self._moment_condition_i(j1, j2, 0).shape[0]
        # print(tmp)
        Omega_hat = np.zeros((tmp, tmp))
        for i in range(self.n):
            moment_condition = self._moment_condition_i(j1, j2, i)
            Omega_hat_i = moment_condition @ moment_condition.T
            Omega_hat += Omega_hat_i
        Omega_hat = Omega_hat/self.n

        return Omega_hat
    

    def _calc_Omega_hat_phase_two(self, j1, j2, sigma):
        tmp = self._moment_condition_i_phase_two(j1, j2, 0, sigma).shape[0]
        Omega_hat = np.zeros((tmp, tmp))
        for i in range(self.n):
            moment_condition = self._moment_condition_i_phase_two(j1, j2, i, sigma)
            Omega_hat_i = moment_condition @ moment_condition.T
            Omega_hat += Omega_hat_i
        Omega_hat = Omega_hat/self.n

        return Omega_hat
    
    
    # Look for the dimensions whose eigenvalue is small
    def _recheck_omega_hat_invertibility(self, j1, j2):
        Omega_hat = self._calc_Omega_hat(j1,j2)
        eigenvalues, _ = np.linalg.eig(Omega_hat)
        anomaly_list = []
        idx = 0
        for eigenvalue in eigenvalues:
            if eigenvalue < 1e-5:
                anomaly_list.append(idx)
            idx = idx+1 
        
        self.block_indx_matrix[j1][j2] = [self.block_indx_matrix[j1][j2][i] for i in range(len(self.block_indx_matrix[j1][j2])) if i not in anomaly_list]
        self.boundaries_matrix[j1][j2] = [self.boundaries_matrix[j1][j2][i] for i in range(len(self.boundaries_matrix[j1][j2])) if i not in anomaly_list]
        self.tau_matrix[j1][j2] = np.delete(self.tau_matrix[j1][j2], anomaly_list)


    def block_probability(self, b_X1, b_X2, sigma, block_idx):
        # Extract block indices
        i, j = block_idx
        
        # Define the extended boundaries with infinities
        b_X1_ext = np.concatenate(([-np.inf], b_X1, [np.inf]))
        b_X2_ext = np.concatenate(([-np.inf], b_X2, [np.inf]))
        
        # Get lower and upper bounds for the block
        x_lower = b_X1_ext[i]
        x_upper = b_X1_ext[i+1]
        y_lower = b_X2_ext[j]
        y_upper = b_X2_ext[j+1]
        
        # Define mean and covariance matrix
        mean = [0, 0]
        cov = [[1, sigma], [sigma, 1]]
        
        # Compute the probability over the block
        upper = [x_upper, y_upper]
        lower = [x_lower, y_lower]
        prob, _ = mvn.mvnun(lower, upper, mean, cov)
        return prob


    def estimate_gradient(self, b_X1, b_X2, sigma, block_idx, h=1e-5):
        # Flatten variables into a single array
        theta = np.concatenate(([sigma], b_X1, b_X2))
        grad = np.zeros_like(theta)
        num_vars = len(theta)
        
        # Compute the original probability
        # P0 = self.block_probability(b_X1, b_X2, sigma, block_idx)
        
        for k in range(num_vars):
            # Perturbation vector
            e = np.zeros(num_vars)
            e[k] = h
            
            # Perturb variables
            theta_plus = theta + e
            theta_minus = theta - e
            
            # Split variables back
            sigma_plus = theta_plus[0]
            b_X1_plus = theta_plus[1:len(b_X1)+1]
            b_X2_plus = theta_plus[len(b_X1)+1:]
            
            sigma_minus = theta_minus[0]
            b_X1_minus = theta_minus[1:len(b_X1)+1]
            b_X2_minus = theta_minus[len(b_X1)+1:]
            
            
            # Compute perturbed probabilities
            P_plus = self.block_probability(b_X1_plus, b_X2_plus, sigma_plus, block_idx)
            P_minus = self.block_probability(b_X1_minus, b_X2_minus, sigma_minus, block_idx)
            
            # Estimate gradient
            grad[k] = (P_plus - P_minus) / (2 * h)
        
        return -grad


    def estimate_gradient_single(self, b_X1, b_X2, sigma, block_idx, h=1e-5):
        # Flatten variables into a single array
        theta = np.concatenate(([sigma], b_X1))
        grad = np.zeros_like(theta)
        num_vars = len(theta)
        
        for k in range(num_vars):
            # Perturbation vector
            e = np.zeros(num_vars)
            e[k] = h
            
            # Perturb variables
            theta_plus = theta + e
            theta_minus = theta - e
            
            # Split variables back
            sigma_plus = theta_plus[0]
            b_X1_plus = theta_plus[1:len(b_X1)+1]
            
            sigma_minus = theta_minus[0]
            b_X1_minus = theta_minus[1:len(b_X1)+1]
            
            
            # Compute perturbed probabilities
            P_plus = self.block_probability(b_X1_plus, b_X2, sigma_plus, block_idx)
            P_minus = self.block_probability(b_X1_minus, b_X2, sigma_minus, block_idx)
            
            # Estimate gradient
            grad[k] = (P_plus - P_minus) / (2 * h)
        
        
        return -grad



class TwoStepGMM():
    def __init__(self, X_obs):
        self.X = X_obs.copy()
        self.n = self.X.shape[0]
        self.p = self.X.shape[1]
        self.est = Estimation(self.X)
        self.Sigma_phase_one = self._calc_Sigma_phase_one()


    def _calc_Sigma_phase_one(self):
        Sigma = np.zeros((self.p, self.p))
        for j1 in range(self.p):
            for j2 in range(self.p):
                Sigma[j1, j2] = self.est._calc_sigma_phase_one_chosen_blocks(j1, j2)
        return Sigma
    

    def _calc_grad(self, j1, j2, sigma):
        h_hat_j1 = self.est._calc_h_hat(j1)
        h_hat_j2 = self.est._calc_h_hat(j2)
        block_idxs = self.est.block_indx_matrix[j1][j2]
        if j1 == j2:
            grads = np.zeros((len(block_idxs), len(h_hat_j1) + 1))
        else:
            grads = np.zeros((len(block_idxs), len(h_hat_j1) + len(h_hat_j2) + 1))
            
        idx = 0
        for block_idx in block_idxs:
            if j1 == j2:
                grad = self.est.estimate_gradient_single(h_hat_j1, h_hat_j2, sigma, block_idx)
            else:
                grad = self.est.estimate_gradient(h_hat_j1, h_hat_j2, sigma, block_idx)
            grads[idx] = grad
            idx += 1

        return grads


    # def _calc_grad(self, j1, j2, sigma):
    #     h_hat_j1 = self.est._calc_h_hat(j1)
    #     h_hat_j2 = self.est._calc_h_hat(j2)
    #     block_idxs = self.est.block_indx_matrix[j1][j2]
    #     grads = np.zeros((len(block_idxs), len(h_hat_j1) + len(h_hat_j2) + 1))
    #     idx = 0
    #     for block_idx in block_idxs:
    #         grad = self.est.estimate_gradient(h_hat_j1, h_hat_j2, sigma, block_idx)
    #         grads[idx] = grad
    #         idx += 1

    #     return grads
    
    def _inference_two_step(self, j1, j2):
        Omega_hat = self.est._calc_Omega_hat(j1, j2)
        A = np.linalg.inv(Omega_hat)
        sigma = self.est._calc_sigma_phase_two(j1, j2, A)
        Omega_hat_phase_two = self.est._calc_Omega_hat_phase_two(j1, j2, sigma)
        A_phase_two = np.linalg.inv(Omega_hat_phase_two)
        G = self._calc_grad(j1, j2, sigma)
        variance = np.linalg.inv(G.T @ A_phase_two @ G)

        sigma_tested = sigma
        z_score = (sigma_tested*np.sqrt(self.n) - 0)/np.sqrt(variance[0,0])
        p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
        
        print("Sigma:", sigma_tested)
        print("Variance:", variance)
        print("P-value:", p_value)  

        return p_value
        

    