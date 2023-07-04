import numpy as np
import scipy.sparse as sp
import torch

import ABIDEParser as Reader
from sklearn.model_selection import StratifiedKFold
from scipy.spatial import distance
from scipy.sparse.linalg.eigen.arpack import eigsh


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)

def get_train_test_masks(labels, idx_train, idx_val, idx_test):
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return y_train, y_val, y_test, train_mask, val_mask, test_mask

def load_data(subject_IDs, params): 
    
    # labels
    num_classes = 2
    num_nodes = len(subject_IDs)
    
    # 初始化y_data(), y
    y_data = np.zeros([num_nodes, num_classes])
    y = np.zeros([num_nodes, 1])
    
    labels = Reader.get_subject_score(subject_IDs, score='DX_GROUP')
    features = Reader.get_networks(subject_IDs, kind=params['connectivity'], atlas_name=params['atlas'])
    
    for i in range(num_nodes):
        y_data[i, int(labels[subject_IDs[i]]) - 1] = 1 # (871,2)
        y[i] = int(labels[subject_IDs[i]]) # (871,)
        
    skf = StratifiedKFold(n_splits=10)
    cv_splits = list(skf.split(features, np.squeeze(y)))
    train = cv_splits[params['folds']][0]
    test = cv_splits[params['folds']][1]
    val = test
    
    print('Number of train sample:{}' .format(len(train)))
        
    y_train, y_val, y_test, train_mask, val_mask, test_mask = get_train_test_masks(y_data, train, val, test)
    
    y_data = torch.LongTensor(np.where(y_data)[1])
    y = torch.LongTensor(y)
    y_train = torch.LongTensor(y_train[1])
    y_val = torch.LongTensor(y_val[1])
    y_test = torch.LongTensor(y_test[1])
    
    train = torch.LongTensor(train)
    val = torch.LongTensor(val)
    test = torch.LongTensor(test)
    train_mask = torch.LongTensor(train_mask)
    val_mask = torch.LongTensor(val_mask)
    test_mask = torch.LongTensor(test_mask)
    
    # Eigenvector
    labeled_ind = Reader.site_percentage(train, params['num_training'], subject_IDs)
    x_data = Reader.feature_selection(features, y, labeled_ind, params['num_features'])
    features = preprocess_features(sp.coo_matrix(x_data).tolil())
    features = torch.FloatTensor(np.array(features.todense()))
    
    # Adjacency matrix
    graph = Reader.create_affinity_graph_from_scores(['SEX', 'SITE_ID'], subject_IDs)
    distv = distance.pdist(x_data, metric='correlation')
    dist = distance.squareform(distv)
    sigma = np.mean(dist)
    sparse_graph = np.exp(- dist ** 2 / (2 * sigma ** 2))
    final_graph = graph * sparse_graph

    return final_graph, features, y, y_data, y_train, y_val, y_test, train, val, test, train_mask, val_mask, test_mask


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        coords = torch.from_numpy(coords)
        values = torch.from_numpy(values)
        shape = torch.tensor(shape)
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features

def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return adj_normalized

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return t_k

