import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot


def load_data(path="../data/eng/", dataset="knowledge_points"):
    """Load knowledge points dataset"""
    print('Loading {} dataset...'.format(dataset))

    # Load features and labels
    data = np.genfromtxt("{}{}.points".format(path, dataset), dtype=np.float32)
    features = sp.csr_matrix(data[:, 1:], dtype=np.float32)  # All columns except the first are features
    labels = data[:, 0].astype(np.int32)  # First column is the course ID

    # Build graph
    edges_unordered = np.genfromtxt("{}{}.links".format(path, dataset), dtype=np.int32)
    num_nodes = features.shape[0]
    # Filter edges with node indices exceeding the number of nodes
    edges_unordered = edges_unordered[(edges_unordered[:, 0] < num_nodes) & (edges_unordered[:, 1] < num_nodes)]
    adj = sp.coo_matrix((np.ones(edges_unordered.shape[0]), (edges_unordered[:, 0], edges_unordered[:, 1])),
                        shape=(num_nodes, num_nodes),
                        dtype=np.float32)

    # Build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)  # Adjust these indices as needed
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)
