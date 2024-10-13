import os.path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import DataSet
from utils import BPRLoss, EmbLoss


class SpAdjEdgeDrop(nn.Module):
    def __init__(self):
        super(SpAdjEdgeDrop, self).__init__()

    def forward(self, adj, keep_rate):
        if keep_rate == 1.0:
            return adj
        vals = adj._values()
        idxs = adj._indices()
        edge_num = vals.size()
        mask = (torch.rand(edge_num) + keep_rate).floor().type(torch.bool)
        new_vals = vals[mask]
        new_idxs = idxs[:, mask]
        return torch.sparse.FloatTensor(new_idxs, new_vals, adj.shape)


class GCN(nn.Module):
    def __init__(self, layers, args):
        super(GCN, self).__init__()
        self.layers = layers
        self.edge_dropper = SpAdjEdgeDrop()
        self.embedding_size = args.embedding_size
        self.gcn_method = args.gcn_method
        self.if_add_weight, self.if_add_bias = args.if_add_weight, args.if_add_bias
        if self.if_add_weight:
            self.linear_layers = nn.ModuleList(
                [nn.Linear(self.embedding_size, self.embedding_size, self.if_add_bias) for i in range(self.layers)]
            )

    def forward(self, x, adj, keep_rate):
        all_embeddings = [x]
        for i in range(self.layers):
            _adj = self.edge_dropper(adj, keep_rate)
            x = torch.sparse.mm(_adj, x)
            all_embeddings.append(x)
        if self.gcn_method == 'mean':
            x = torch.mean(torch.stack(all_embeddings, dim=0), dim=0)
        return x


class DAGCN(nn.Module):
    def __init__(self, args, dataset: DataSet):
        super(DAGCN, self).__init__()
        self.device = args.device
        self.model_path = args.model_path
        self.checkpoint = args.checkpoint
        self.if_load_model = args.if_load_model

        # Base
        self.n_users = dataset.user_count
        self.n_items = dataset.item_count
        self.embedding_size = args.embedding_size
        self.user_embedding = nn.Embedding(self.n_users + 1, self.embedding_size, padding_idx=0)
        self.item_embedding = nn.Embedding(self.n_items + 1, self.embedding_size, padding_idx=0)

        # Modules
        self.behaviors = dataset.behaviors
        self.keep_rate = args.keep_rate
        self.if_layer_norm = args.if_layer_norm
        self.behavior_adjs = dataset.behavior_adjs

        self.bpr_loss = BPRLoss()
        self.emb_loss = EmbLoss()
        self.pre_behavior_dict = dataset.pre_behavior_dict
        self.behavior_cf_layers = dataset.behavior_cf_layers
        self.personal_trans_dict = dataset.personal_trans_dict

        self.behavior_cf = defaultdict(dict)
        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            pre_behavior_cf = nn.ModuleDict()
            for pre_beh in pre_behaviors:
                pre_behavior_cf[pre_beh] = GCN(self.behavior_cf_layers[post_beh][pre_beh], args)
            self.behavior_cf[post_beh] = pre_behavior_cf

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            for pre_beh in pre_behaviors:
                trans_mat = self.personal_trans_dict[post_beh][pre_beh].detach()
                self.personal_trans_dict[post_beh][pre_beh] = nn.Parameter(trans_mat, requires_grad=False).to(
                    self.device)

        # Loss
        self.reg_weight = args.reg_weight
        self.aux_weight = args.aux_weight
        self.if_multi_tasks = args.if_multi_tasks
        self.mtl_type = args.mtl_type
        self.personal_loss_ratios = dataset.personal_loss_ratios
        self.global_loss_ratios = dataset.global_loss_ratios
        self.loss_ratio_type = args.loss_ratio_type

        self.storage_all_embeddings = None

        self._init_weights()
        self._load_model()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight.data)
        nn.init.xavier_uniform_(self.item_embedding.weight.data)

    def _load_model(self):
        if self.if_load_model:
            parameters = torch.load(os.path.join(self.model_path, self.checkpoint, 'model.pth'))
            self.load_state_dict(parameters, strict=False)

    def gcn_propagate(self):
        all_embeddings = {}
        last_embedding = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings['all'] = last_embedding

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            pre_behaviors = pre_behaviors[::-1]
            post_embeddings = []
            for index, pre_beh in enumerate(pre_behaviors):
                pre_embedding = all_embeddings[pre_beh]
                layer_adj = self.behavior_adjs[post_beh].to(self.device)
                lightgcn_all_embeddings = self.behavior_cf[post_beh][pre_beh](pre_embedding, layer_adj, self.keep_rate)
                trans_mat = self.personal_trans_dict[post_beh][pre_beh].to(self.device)
                post_embedding = torch.mul(trans_mat, lightgcn_all_embeddings)
                post_embeddings.append(post_embedding)
            agg_messages = sum(post_embeddings)
            if self.if_layer_norm:
                agg_messages = F.normalize(agg_messages, dim=-1)
            cur_embedding = agg_messages + last_embedding
            all_embeddings[post_beh] = cur_embedding
            last_embedding = cur_embedding
        return all_embeddings

    def forward(self, batch_data):
        self.storage_all_embeddings = None

        all_embeddings = self.gcn_propagate()  # dict (|B|, N, dim)
        total_loss = 0
        for index, behavior in enumerate(self.behaviors):
            if self.if_multi_tasks or behavior == self.behaviors[-1]:
                data = batch_data[:, index]  # (bsz,3)
                users = data[:, 0].long()  # (bsz,)
                items = data[:, 1:].long()  # (bsz, 2)
                user_all_embedding, item_all_embedding = torch.split(all_embeddings[behavior],
                                                                     [self.n_users + 1, self.n_items + 1])

                user_feature = user_all_embedding[users.view(-1, 1)].expand(-1, items.shape[1], -1)  # (bsz, 2, dim)
                item_feature = item_all_embedding[items]  # (bsz, 2, dim)
                scores = torch.sum(user_feature * item_feature, dim=2)  # (bsz, 2)

                mask = torch.where(users != 0)[0]
                scores = scores[mask]

                # MTL - Personalized
                if self.mtl_type == 'personalized':
                    if behavior == self.behaviors[-1]:
                        user_loss_ratios = torch.ones_like(users).float()
                    else:
                        user_loss_ratios = self.personal_loss_ratios[behavior][users].to(self.device)
                        user_loss_ratios = self.aux_weight * user_loss_ratios
                    user_loss_ratios = user_loss_ratios[mask]
                    total_loss += (user_loss_ratios * self.bpr_loss(scores[:, 0], scores[:, 1])).mean()

                # MTL - Global
                elif self.mtl_type == 'global':
                    if behavior == self.behaviors[-1]:
                        beh_loss_ratio = 1.0
                    else:
                        beh_loss_ratio = self.aux_weight * self.global_loss_ratios[behavior]
                    total_loss += (beh_loss_ratio * self.bpr_loss(scores[:, 0], scores[:, 1])).mean()

                # Single Task or MTL-addition
                else:
                    total_loss += (self.bpr_loss(scores[:, 0], scores[:, 1])).mean()

        total_loss = total_loss + self.reg_weight * self.emb_loss(self.user_embedding.weight,
                                                                  self.item_embedding.weight)

        return total_loss

    def full_predict(self, users):
        if self.storage_all_embeddings is None:
            self.storage_all_embeddings = self.gcn_propagate()

        user_embedding, item_embedding = torch.split(self.storage_all_embeddings[self.behaviors[-1]],
                                                     [self.n_users + 1, self.n_items + 1])
        user_emb = user_embedding[users.long()]  # (test_bsz, dim)
        scores = torch.matmul(user_emb, item_embedding.transpose(0, 1))  # (test_bsz, |I|)
        return scores
