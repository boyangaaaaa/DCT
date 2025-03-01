import numpy as np
from scipy.special import expit as sigmoid
import igraph as ig
import random
import causaldag as cd

def get_skeleton(B):
    B_bin = (B != 0).astype(int)
    return ((B_bin + B_bin.T) != 0).astype(int)


def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


def get_cpdag_from_cdnod(g):
    a,b = g.shape
    cpdag = np.zeros((a,b))
    for i in range(a):
        for j in range(b):
            if g[j,i]==1 and g[i,j]==-1:
                cpdag[i,j] = 1
            elif g[j,i]==-1 and g[i,j]==1:
                cpdag[j,i] = 1
            elif g[j,i]==-1 and g[i,j]==-1:
                cpdag[j,i] = 1
                cpdag[i,j] = 1
            # elif g[j,i]==1 and g[i,j]==1:
            #     bin_g[j,i] = 1   
    return cpdag


def get_dag_from_pdag(B_bin_pdag):
    # There is bug for G.to_dag().to_amat() from cd package
    # i.e., the shape of B is not preserved
    # So we need to manually preserve the shape
    B_bin_dag = np.zeros_like(B_bin_pdag)
    if np.all(B_bin_pdag == 0):
        # All entries in B_pdag are zeros
        return B_bin_dag
    else:
        G = cd.PDAG.from_amat(B_bin_pdag)  # return a PDAG with arcs/edges.
        # print(G.to_amat()[0])
        B_bin_sub_dag, nodes = G.to_dag().to_amat() # The key is: to_dag() - converting a PDAG to a DAG using some rules. 
        # print("G:", G.to_dag().to_amat())
        B_bin_dag[np.ix_(nodes, nodes)] = B_bin_sub_dag
        return B_bin_dag


def is_dag(W):
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    return G.is_dag()


def simulate_linear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from linear SEM with specified type of noise.

    For uniform, noise z ~ uniform(-a, a), where a = noise_scale.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples, n=inf mimics population risk
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix, [d, d] if n=inf
    """ 
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            mean = np.random.uniform(-2,2, size=1)
            z = np.random.normal(loc=mean, scale=scale, size=n) # loc=mean, scale=std.
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif sem_type == 'student':
            z = np.random.standard_t(df=3, size=n)
            x = X @ w + z
        elif sem_type == 'mix_gauss':
            z1 = np.random.normal(loc=0, scale=scale, size=int(n/2))
            z2 = np.random.normal(loc=2, scale=scale, size=int(n/2))
            z = np.concatenate((z1, z2))
            x = X @ w + z
        else:
            raise ValueError('unknown sem type')
        return x

    d = W.shape[0]
    b = np.random.uniform(low=0.5, high=2.5, size=(d,d))
    W = b * W
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        # print(f"Parent: {parents+np.ones(len(parents))}. child:{j+1}.")
    return X


def simulate_nonlinear_sem(W, n, sem_type, noise_scale=None):
    """Simulate samples from nonlinear SEM with specified type of noise.

    Args:
        W (np.ndarray): [d, d] weighted adj matrix of DAG
        n (int): num of samples
        sem_type (str): gauss, exp, gumbel, uniform, logistic, poisson
        noise_scale (np.ndarray): scale parameter of additive noise, default all ones

    Returns:
        X (np.ndarray): [n, d] sample matrix
    """
    def _simulate_single_equation(X, w, scale):
        """X: [n, num of parents], w: [num of parents], x: [n]"""
        if sem_type == 'gauss':
            z = np.random.normal(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'exp':
            z = np.random.exponential(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'gumbel':
            z = np.random.gumbel(scale=scale, size=n)
            x = X @ w + z
        elif sem_type == 'uniform':
            z = np.random.uniform(low=-scale, high=scale, size=n)
            x = X @ w + z
        elif sem_type == 'logistic':
            x = np.random.binomial(1, sigmoid(X @ w)) * 1.0
        elif sem_type == 'poisson':
            x = np.random.poisson(np.exp(X @ w)) * 1.0
        elif sem_type == 'student':
            z = np.random.standard_t(df=3, size=n)
            x = X @ w + z
        elif sem_type == 'mix_gauss':
            z1 = np.random.normal(loc=0, scale=scale, size=int(n/2))
            z2 = np.random.normal(loc=2, scale=scale, size=int(n/2))
            z = np.concatenate((z1, z2))
            x = X @ w + z
        else:
            raise ValueError('unknown sem type')
        return x
    
    def f(x):
        det = np.random.randint(4) 
        if det == 0:
            y = np.sin(x)
        elif det == 1:
            y = x**2
        elif det == 2:
            y = np.tanh(x)
        elif det == 3:
            y = np.maximum(x,0)
        return y 
    
    d = W.shape[0]
    b = np.random.uniform(low=0.5, high=2.5, size=(d,d))
    W = b * W
    if noise_scale is None:
        scale_vec = np.ones(d)
    elif np.isscalar(noise_scale):
        scale_vec = noise_scale * np.ones(d)
    else:
        if len(noise_scale) != d:
            raise ValueError('noise scale must be a scalar or has length d')
        scale_vec = noise_scale
    if not is_dag(W):
        raise ValueError('W must be a DAG')
        
    if np.isinf(n):  # population risk for linear gauss SEM
        if sem_type == 'gauss':
            # make 1/d X'X = true cov
            X = np.sqrt(d) * np.diag(scale_vec) @ np.linalg.inv(np.eye(d) - W)
            return X
        else:
            raise ValueError('population risk not available')
        
    # empirical risk
    G = ig.Graph.Weighted_Adjacency(W.tolist())
    ordered_vertices = G.topological_sorting()
    assert len(ordered_vertices) == d
    X = np.zeros([n, d])    
    for j in ordered_vertices:
        parents = G.neighbors(j, mode=ig.IN)
        X[:, j] = _simulate_single_equation(X[:, parents], W[parents, j], scale_vec[j])
        X[:, j] = f(X[:,j])
        # print(f"Parent: {parents+np.ones(len(parents))}. child:{j+1}.")
    return X    
    
    


def simulate_dag(d, s0, graph_type):
    """Simulate random DAG with some expected number of edges.

    Args:
        d (int): num of nodes
        s0 (int): expected num of edges
        graph_type (str): ER, SF, BP

    Returns:
        B (np.ndarray): [d, d] binary adj matrix of DAG
    """
    def _random_permutation(M):
        # np.random.permutation permutes first axis only
        P = np.random.permutation(np.eye(M.shape[0]))
        return P.T @ M @ P

    def _random_acyclic_orientation(B_und):
        return np.tril(_random_permutation(B_und), k=-1)

    def _graph_to_adjmat(G):
        return np.array(G.get_adjacency().data)

    if graph_type == 'ER':
        # Erdos-Renyi
        G_und = ig.Graph.Erdos_Renyi(n=d, m=s0)#m=d, p=0.3 p=0.5
        B_und = _graph_to_adjmat(G_und)
        B = _random_acyclic_orientation(B_und)
    elif graph_type == 'SF':
        # Scale-free, Barabasi-Albert
        G = ig.Graph.Barabasi(n=d, m=int(round(s0 / d)) + 1, directed=True)
        B = _graph_to_adjmat(G)
    elif graph_type == 'BP':
        # Bipartite, Sec 4.1 of (Gu, Fu, Zhou, 2018)
        top = np.ceil(0.4 * d).astype(int)
        print(top)
        G = ig.Graph.Random_Bipartite(top, d - top, m=s0, directed=True, neimode=ig.OUT)
        B = _graph_to_adjmat(G)
    else:
        raise ValueError('unknown graph type')
    B_perm = _random_permutation(B)
    assert ig.Graph.Adjacency(B_perm.tolist()).is_dag()
    return B_perm


def _binary(X_true, rand_integer):
    
    X = X_true.copy()
    for i in rand_integer:
        x_tmp = X[:,i]
        var = np.var(x_tmp)
        c = np.random.uniform(-var/2, var/2, 1).item()
        print(f"Random variable: {i}, threshold: {c}")
        # x_tmp[x_tmp < c] = 0
        # x_tmp[x_tmp > c] = 1
        x_tmp = np.where(x_tmp > c, 1, 0)
        X[:,i] = x_tmp
    return X


def _binary_all(X_true):
    X = X_true.copy()
    for i in range(X.shape[1]):
        x_tmp = X[:,i]
        var = np.var(x_tmp)
        # c = np.random.uniform(-var/8, var/8, 1).item()
        c = np.median(x_tmp)
        print(f"Random variable: {i}, threshold: {c}")
        # x_tmp[x_tmp < c] = 0
        # x_tmp[x_tmp > c] = 1
        x_tmp = np.where(x_tmp > c, 1, 0)
        X[:,i] = x_tmp
    return X


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


def covariance_to_correlation(cov_matrix):
    """
    Convert a covariance matrix to a correlation matrix.

    Parameters:
    cov_matrix (numpy.ndarray): A square covariance matrix.

    Returns:
    numpy.ndarray: The corresponding correlation matrix.
    """
    if cov_matrix.shape[0] != cov_matrix.shape[1]:
        raise ValueError("Covariance matrix should be square.")

    # Standard deviations (sqrt of diagonal elements of the covariance matrix)
    std_devs = np.sqrt(np.diag(cov_matrix))

    # Outer product of standard deviations
    std_devs_matrix = np.outer(std_devs, std_devs)

    # Correlation matrix
    corr_matrix = cov_matrix / std_devs_matrix

    # Setting the diagonal elements to 1
    np.fill_diagonal(corr_matrix, 1)

    return corr_matrix


def count_skeleton_accuracy(B_bin_true, B_bin_est):
    skeleton_true = get_skeleton(B_bin_true) # b_bin_true[i,j]=1  <==> skeleton[i,j]=skeleton[j,i]=1
    skeleton_est = get_skeleton(B_bin_est)   # b_bin_est[i,j]=-1 & b_bin_est[j,i]=1  <==>  skeleton[i,j]=skeleton[j,i]=1

    # print(3, skeleton_true)
    # print(4, skeleton_est) 

    d = len(skeleton_true)
    skeleton_triu_true = skeleton_true[np.triu_indices(d, k=1)]
    skeleton_triu_est = skeleton_est[np.triu_indices(d, k=1)]
    pred = np.flatnonzero(skeleton_triu_est)  # estimated graph
    cond = np.flatnonzero(skeleton_triu_true) # true graph 

    # true pos: an edge estimated with correct direction.
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos: an edge that is in estimated graph but not in the true graph.
    false_pos = np.setdiff1d(pred, cond, assume_unique=True)
    # false neg: an edge that is not in estimated graph but in the true graph.
    false_neg = np.setdiff1d(cond, pred, assume_unique=True) # This is also OK: np.setdiff1d(cond, true_pos, assume_unique=True)
    # true negative: an edge that is neither in estimated graph nor in true graph.
    # true negative: normally equals 0.

    # compute ratio
    nnz = len(pred)
    cond_neg_size = len(skeleton_triu_true) - len(cond)
    fdr = float(len(false_pos)) / max(nnz, 1)  # fdr = (FP) / (TP + FP) = FP / |pred_graph|
    tpr = float(len(true_pos)) / max(len(cond), 1)  # tpr: TP / (TP + FN) = TP / |true_graph|
    fpr = float(len(false_pos)) / max(cond_neg_size, 1) # fpr: (FP) / (TN + FP) = FP / ||
    try:
        f1 = len(true_pos) / (len(true_pos) + 0.5 * (len(false_pos) + len(false_neg)))
    except:
        f1 = None

    # structural hamming distance
    extra_lower = np.setdiff1d(pred, cond, assume_unique=True)
    missing_lower = np.setdiff1d(cond, pred, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower)
    return {'f1_skeleton': f1, 'precision_skeleton': 1 - fdr, 'recall_skeleton': tpr, 'shd_skeleton': shd}
    # return {'f1_skeleton': f1, 'precision_skeleton': 1 - fdr, 'recall_skeleton': tpr,
            # 'shd_skeleton': shd, 'TPR_skeleton': tpr, 'FDR_skeleton': fdr, "number_edge_pred":len(pred), "number_edge_true":len(cond)}



def count_dag_accuracy(B_bin_true, B_bin_est):
    print('Hi, this is count dag accuracy')
    print(B_bin_true)
    print(B_bin_est)

    d = B_bin_true.shape[0]
    # linear index of nonzeros
    pred = np.flatnonzero(B_bin_est)
    cond = np.flatnonzero(B_bin_true)
    cond_reversed = np.flatnonzero(B_bin_true.T)
    cond_skeleton = np.concatenate([cond, cond_reversed])
    print(pred, cond)
    # true pos
    true_pos = np.intersect1d(pred, cond, assume_unique=True)
    # false pos
    false_pos = np.setdiff1d(pred, cond_skeleton, assume_unique=True)
    # reverse
    extra = np.setdiff1d(pred, cond, assume_unique=True)
    reverse = np.intersect1d(extra, cond_reversed, assume_unique=True)
    # compute ratio
    pred_size = len(pred)
    cond_neg_size = 0.5 * d * (d - 1) - len(cond)
    if pred_size == 0:
        fdr = None
    else:
        fdr = float(len(reverse) + len(false_pos)) / pred_size
    if len(cond) == 0:
        tpr = None
    else:
        tpr = float(len(true_pos)) / len(cond)
    if cond_neg_size == 0:
        fpr = None
    else:
        fpr = float(len(reverse) + len(false_pos)) / cond_neg_size
    # structural hamming distance
    pred_lower = np.flatnonzero(np.tril(B_bin_est + B_bin_est.T))
    cond_lower = np.flatnonzero(np.tril(B_bin_true + B_bin_true.T))
    extra_lower = np.setdiff1d(pred_lower, cond_lower, assume_unique=True)
    missing_lower = np.setdiff1d(cond_lower, pred_lower, assume_unique=True)
    shd = len(extra_lower) + len(missing_lower) + len(reverse)
    # false neg
    false_neg = np.setdiff1d(cond, true_pos, assume_unique=True)
    precision, recall, f1 = count_precision_recall_f1(tp=len(true_pos),
                                                      fp=len(reverse) + len(false_pos),
                                                      fn=len(false_neg))
    # return {'fdr': fdr, 'tpr': tpr, 'fpr': fpr, 'shd': shd, 'nnz': pred_size, 
    #         'precision': precision, 'recall': recall, 'f1': f1}
    return {'f1': f1,  'precision': precision, 'recall': recall, 'shd': shd}



def count_precision_recall_f1(tp, fp, fn):
    # Precision
    if tp + fp == 0:
        precision = None
    else:
        precision = float(tp) / (tp + fp)

    # Recall
    if tp + fn == 0:
        recall = None
    else:
        recall = float(tp) / (tp + fn)

    # F1 score
    if precision is None or recall is None:
        f1 = None
    elif precision == 0 or recall == 0:
        f1 = 0.0
    else:
        f1 = float(2 * precision * recall) / (precision + recall)
    return precision, recall, f1


