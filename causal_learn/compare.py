from causallearn.utils.cit import CIT
import multiprocessing
import numpy as np
import random



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


def perform_test(args):
    n_samples, seed, dependent = args
    # mean = [0, 0, 0, 0]
    # cov = [[1, 0, 0.5, 0], 
    #         [0, 1, 0, 0],
    #         [0.5, 0, 1, 0.7], 
    #         [0, 0, 0.7, 1]  ]   # Example covariance matrix
    # X_true = np.random.multivariate_normal(mean, cov, n_samples)    
    
    np.random.seed(seed)
    x1 = np.random.normal(0,1,1000)
    x2 = 2 * x1 + np.random.normal(0,1,1000)
    x3 = x2 + np.random.normal(0,1,1000)
    X_true = np.array([x1,x2,x3]).T

    C = [random.uniform(-1, 1) for _ in range(3)]
    conditional_independent_data_obs = _binary(C, X_true)
    fisherz = CIT(conditional_independent_data_obs,  'chisq')

    if dependent:
        p_value = fisherz(0, 1, [2])
    else:
        p_value = fisherz(0, 2, [1])

    return p_value


def main():
    n_samples = 1000
    times = 8000

    # print('The code is running')
    # seeds = range(times)
    # p_value_list = Parallel(n_jobs=1)(delayed(perform_test)(n_samples, seed) for seed in seeds)

    print('The code is running')
    with multiprocessing.Pool() as pool:
        args = [(n_samples, seed, False) for seed in range(times)]
        p_value_list = pool.map(perform_test, args)

    print(len(p_value_list))    
    type_i_error = sum(p_value < 0.05 for p_value in p_value_list) / times
    print('Type I error:', type_i_error)
    np.save('fisherz_p_value_ci' + 'n_samples' + str(n_samples) + 'n_tests' + str(times) +'.npy', p_value_list)


    print('The code is running')
    with multiprocessing.Pool() as pool:
        args = [(n_samples, seed, True) for seed in range(times)]
        p_value_list = pool.map(perform_test, args)

    print(len(p_value_list))    
    power = sum(p_value < 0.05 for p_value in p_value_list) / times
    print('Power is:', power)
    np.save('fishzer_p_value_cd'+ 'n_samples' + str(n_samples) + 'n_tests' + str(times) +'.npy', p_value_list)


if __name__ == "__main__":
    main()