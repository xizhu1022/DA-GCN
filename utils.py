import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import random


class BPRLoss(nn.Module):
    def __init__(self):
        super(BPRLoss, self).__init__()
        self.gamma = 1e-10

    def forward(self, p_score, n_score):
        loss = -torch.log(self.gamma + torch.sigmoid(p_score - n_score))
        return loss


class EmbLoss(nn.Module):
    """ EmbLoss, regularization on embeddings

    """

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings):
        emb_loss = torch.zeros(1).to(embeddings[-1].device)
        for embedding in embeddings:
            emb_loss += torch.norm(embedding, p=self.norm)
        emb_loss /= embeddings[-1].shape[0]
        return emb_loss


class InfoNCELoss(nn.Module):
    """
    From SSLRec models/loss.utils
    """

    def __init__(self, temp=1.0):
        super(InfoNCELoss, self).__init__()
        self.temp = temp

    def forward(self, embeds1, embeds2, all_embeds2):
        normed_embeds1 = embeds1 / torch.sqrt(1e-8 + embeds1.square().sum(-1, keepdim=True))
        normed_embeds2 = embeds2 / torch.sqrt(1e-8 + embeds2.square().sum(-1, keepdim=True))
        normed_all_embeds2 = all_embeds2 / torch.sqrt(1e-8 + all_embeds2.square().sum(-1, keepdim=True))

        nume_term = - (normed_embeds1 * normed_embeds2 / self.temp).sum(-1)
        deno_term = torch.log(torch.sum(torch.exp(normed_embeds1 @ normed_all_embeds2.T / self.temp), dim=-1))
        cl_loss = (nume_term + deno_term).sum()

        return cl_loss


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_adj(adj, method='asym'):
    if method == 'sym':
        degree = np.array(adj.sum(axis=-1))
        d_inv_sqrt = np.reshape(np.power(degree, -0.5), [-1])
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
        d_inv_sqrt_mat = sp.diags(d_inv_sqrt)
        norm_adj = d_inv_sqrt_mat.dot(adj).dot(d_inv_sqrt_mat)

    elif method == 'asym':
        degree = np.array(adj.sum(axis=-1))
        d_inv = np.reshape(np.power(degree, -1), [-1])
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv_mat = sp.diags(d_inv)
        norm_adj = d_inv_mat.dot(adj)

    elif method == 'mean':
        degree = np.array(adj.sum(axis=-1))
        d_inv = np.reshape(np.power(degree, -1), [-1])
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv_mat = sp.diags(d_inv)
        norm_adj = adj.dot(d_inv_mat)
    else:
        norm_adj = adj
    return norm_adj.tocoo()


def _get_static_hyper_adj(inc_mat: sp.dok_matrix):
    edge_count = inc_mat.shape[1]
    edge_weight = sp.diags(np.ones(edge_count))
    dv = np.array((inc_mat * edge_weight).sum(axis=1))
    de = np.array(inc_mat.sum(axis=0))

    de_inv = np.reshape(np.power(de, -1), [-1])
    de_inv[np.isinf(de_inv)] = 0.0
    de_inv_mat = sp.diags(de_inv)
    inc_mat_transpose = inc_mat.transpose()

    dv_inv_sqrt = np.reshape(np.power(dv, -0.5), [-1])
    dv_inv_sqrt[np.isinf(dv_inv_sqrt)] = 0.0
    dv_inv_sqrt_mat = sp.diags(dv_inv_sqrt)

    g = dv_inv_sqrt_mat * inc_mat * edge_weight * de_inv_mat * inc_mat_transpose * dv_inv_sqrt_mat
    return g


def sp_mat_to_torch_sp_tensor(mat):
    idxs = torch.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
    vals = torch.from_numpy(mat.data.astype(np.float32))
    shape = torch.Size(mat.shape)
    return torch.sparse.FloatTensor(idxs, vals, shape)


def dict2set(_dict):
    _set = set()
    for k, v in _dict.items():
        for _v in v:
            _set.add((k, _v))
    return _set
