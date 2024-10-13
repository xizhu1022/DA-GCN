import json
import os
import random
from collections import defaultdict

import numpy as np
import scipy.sparse as sp
import torch
from loguru import logger
from torch.utils.data import Dataset

from utils import sp_mat_to_torch_sp_tensor, normalize_adj, dict2set


class MBTestData(Dataset):
    def __init__(self, user_count, item_count, samples=None):
        self.user_count = user_count
        self.item_count = item_count
        self.samples = samples

    def __getitem__(self, idx):
        return int(self.samples[idx])

    def __len__(self):
        return len(self.samples)


class MBTrainData(Dataset):
    def __init__(self, user_count, item_count, behavior_dict=None, behaviors=None):
        self.user_count = user_count
        self.item_count = item_count
        self.behavior_dict = behavior_dict
        self.behaviors = behaviors

    def __getitem__(self, idx):
        # generate positive and negative pairs under each behavior
        total = []
        for behavior in self.behaviors:

            items = self.behavior_dict[behavior].get(idx + 1, None)
            if items is None:
                signal = [0, 0, 0]
            else:
                pos = random.sample(items, 1)[0]
                neg = random.randint(1, self.item_count)
                while np.isin(neg, self.behavior_dict['all'][idx + 1]):
                    neg = random.randint(1, self.item_count)
                signal = [idx + 1, pos, neg]
            total.append(signal)
        return np.array(total)

    def __len__(self):
        return self.user_count


class DataSet(object):
    def __init__(self, args):
        self.data_path = args.data_path
        self.data_name = args.data_name
        self.behaviors = args.behaviors
        self.adj_type = args.adj_type
        self.trans_type = args.trans_type
        self.loss_ratio_type = args.loss_ratio_type

        self.train_user_behavior_dict = defaultdict(dict)
        self.train_item_behavior_dict = defaultdict(dict)

        self.behavior_adjs = {}
        self.behavior_mats = {}

        self.get_behaviors()
        self.get_count()
        self.get_behavior_items()
        self.get_behavior_users()
        self.get_validation_dict()
        self.get_test_dict()
        self.get_interaction_mats()
        self.get_cross_beh_info()

        self.user_personal_trans_dict, self.user_global_trans_dict = self.get_pairwise_transfer_ratios(
            train_behavior_dict=self.train_user_behavior_dict, keys=self.users)

        self.item_personal_trans_dict, self.item_global_trans_dict = self.get_pairwise_transfer_ratios(
            train_behavior_dict=self.train_item_behavior_dict, keys=self.items)

        self.personal_trans_dict = self.merge_personal_trans_dict()

        self.personal_loss_ratios, self.global_loss_ratios = self.get_loss_ratios()

        self.validation_gt_length = np.array([len(x) for _, x in self.validation_dict.items()])
        self.test_gt_length = np.array([len(x) for _, x in self.test_dict.items()])

    def get_behaviors(self):
        data2behaviors = {
            'tmall': ['click', 'cart', 'buy'],
            'taobao': ['pv', 'cart', 'buy'],
            'beibei': ['click', 'cart', 'buy'],
            'ijcai': ['click', 'cart', 'buy'],
            'jdata': ['view', 'collect', 'cart', 'buy'],
        }

        if len(self.behaviors) == 0:
            self.behaviors = data2behaviors[self.data_name]

    def get_count(self):
        with open(os.path.join(self.data_path, self.data_name, 'count.txt'), encoding='utf-8') as f:
            count = json.load(f)
            self.user_count = count['user']
            self.item_count = count['item']
            self.users = np.arange(0, self.user_count + 1).tolist()
            self.items = np.arange(0, self.item_count + 1).tolist()
            logger.info('Number of users: {}.'.format(self.user_count))
            logger.info('Number of items: {}.'.format(self.item_count))

    def get_inter_num(self, b_dict):
        return np.sum([len(v) for k, v in b_dict.items()])

    def get_behavior_items(self):
        for behavior in self.behaviors:
            with open(os.path.join(self.data_path, self.data_name, behavior + '_dict.txt'), encoding='utf-8') as f:
                b_dict = json.load(f)
                b_dict = {int(k): v for k, v in b_dict.items()}
                logger.info('Number of {} interactions: {}'.format(behavior, self.get_inter_num(b_dict)))
                self.train_user_behavior_dict[behavior] = b_dict

        all_dict = defaultdict(list)
        for behavior, b_dict in self.train_user_behavior_dict.items():
            for k, v in b_dict.items():
                all_dict[k].extend(v)
        for k, v in all_dict.items():
            items = list(set(v))
            all_dict[k] = sorted(items)
        self.train_user_behavior_dict['all'] = all_dict

    def get_test_dict(self):
        with open(os.path.join(self.data_path, self.data_name, 'test_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            b_dict = {int(k): v for k, v in b_dict.items()}
            self.test_dict = b_dict
            logger.info('Number of test interactions: {}'.format(self.get_inter_num(self.test_dict)))

    def get_validation_dict(self):
        with open(os.path.join(self.data_path, self.data_name, 'validation_dict.txt'), encoding='utf-8') as f:
            b_dict = json.load(f)
            b_dict = {int(k): v for k, v in b_dict.items()}
            self.validation_dict = b_dict
            logger.info('Number of validation interactions: {}'.format(self.get_inter_num(self.validation_dict)))

    def get_cross_beh_info(self):
        behavior_cf_layers = defaultdict(dict)
        full_behaviors = ['all'] + self.behaviors
        self.pre_behavior_dict = {behavior: ['all'] + self.behaviors[:index] for index, behavior in
                                  enumerate(self.behaviors)}
        for post_beh in self.behaviors:
            post_idx = full_behaviors.index(post_beh)
            pre_behaviors = self.pre_behavior_dict[post_beh]
            for pre_beh in pre_behaviors:
                pre_idx = full_behaviors.index(pre_beh)
                behavior_cf_layers[post_beh][pre_beh] = post_idx - pre_idx
        self.behavior_cf_layers = behavior_cf_layers

    def get_behavior_users(self):
        for behavior in self.behaviors + ['all']:
            b_dict = defaultdict(set)
            for user, items in self.train_user_behavior_dict[behavior].items():
                for item in items:
                    b_dict[item].add(user)
            self.train_item_behavior_dict[behavior] = b_dict

    def merge_personal_trans_dict(self):
        personal_trans_dict = defaultdict(dict)
        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            for pre_beh in pre_behaviors:
                trans_mat = torch.cat([self.user_personal_trans_dict[post_beh][pre_beh],
                                       self.item_personal_trans_dict[post_beh][pre_beh]])
                personal_trans_dict[post_beh][pre_beh] = trans_mat
        logger.info('Generate {} transfer ratios'.format(self.trans_type))
        return personal_trans_dict

    def get_pairwise_transfer_ratios(self, train_behavior_dict, keys):
        personal_pw_trans_dict = defaultdict(dict)

        out_personal_pw_trans_dict = defaultdict(dict)
        out_global_pw_trans_dict = defaultdict(dict)

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            pre_behaviors = pre_behaviors[::-1]  # reverse
            pre_beh_trans_dict = defaultdict(dict)
            for key in keys:
                key_trans_dict = {}
                post_items = set(train_behavior_dict[post_beh].get(key, []))
                all_post_items = post_items
                for pre_beh in pre_behaviors:
                    pre_items = set(train_behavior_dict[pre_beh].get(key, []))
                    if self.trans_type == 'post_cond':
                        denominator = all_post_items
                    elif self.trans_type == 'jaccard':
                        denominator = all_post_items | pre_items
                    elif self.trans_type == 'pre_cond':
                        denominator = pre_items
                    else:
                        raise NotImplementedError

                    numerator = post_items & pre_items
                    if len(denominator) == 0:
                        key_trans_dict[pre_beh] = 0.0
                    else:
                        key_trans_dict[pre_beh] = len(numerator) / len(denominator)
                    post_items = post_items - pre_items

                key_trans_sum = np.sum(list(key_trans_dict.values()))
                if key_trans_sum == 0:
                    key_trans_dict = {_k: 0.0 for _k, _v in key_trans_dict.items()}
                    key_trans_dict[pre_behaviors[0]] = 1.0
                else:
                    key_trans_dict = {_k: _v / key_trans_sum for _k, _v in key_trans_dict.items()}
                pre_beh_trans_dict[key] = key_trans_dict
            personal_pw_trans_dict[post_beh] = pre_beh_trans_dict

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            for pre_beh in pre_behaviors:
                trans_mat = [personal_pw_trans_dict[post_beh][key][pre_beh] for key in keys]
                out_personal_pw_trans_dict[post_beh][pre_beh] = torch.FloatTensor(trans_mat).unsqueeze(-1)

        for post_beh in self.behaviors:
            pre_behaviors = self.pre_behavior_dict[post_beh]
            pre_behaviors = pre_behaviors[::-1]

            post_items = dict2set(train_behavior_dict[post_beh])
            all_post_items = post_items
            trans_dict = {}
            for pre_beh in pre_behaviors:
                pre_items = dict2set(train_behavior_dict[pre_beh])
                if self.trans_type == 'post_cond':
                    denominator = all_post_items
                elif self.trans_type == 'jaccard':
                    denominator = all_post_items | pre_items
                elif self.trans_type == 'pre_cond':
                    denominator = pre_items
                else:
                    raise NotImplementedError

                numerator = post_items & pre_items
                trans_dict[pre_beh] = len(numerator) / len(denominator)
                post_items = post_items - pre_items

            trans_sum = np.sum(list(trans_dict.values()))
            trans_dict = {_k: _v / trans_sum for _k, _v in trans_dict.items()}
            out_global_pw_trans_dict[post_beh] = trans_dict

        return out_personal_pw_trans_dict, out_global_pw_trans_dict

    def get_loss_ratios(self):
        personal_loss_ratios = {}
        global_loss_ratios = {}

        for pre_beh in self.behaviors[:-1]:
            loss_ratio = []
            for user in self.users:
                tgt_items = set(self.train_user_behavior_dict[self.behaviors[-1]].get(user, []))
                pre_items = set(self.train_user_behavior_dict[pre_beh].get(user, []))
                if self.loss_ratio_type == 'post_cond':
                    denominator = tgt_items
                elif self.loss_ratio_type == 'jaccard':
                    denominator = tgt_items | pre_items
                elif self.loss_ratio_type == 'pre_cond':
                    denominator = pre_items
                else:
                    raise NotImplementedError

                numerator = tgt_items & pre_items
                if len(denominator) == 0:
                    loss_ratio.append(0.0)
                else:
                    loss_ratio.append(len(numerator) / len(denominator))
            personal_loss_ratios[pre_beh] = torch.FloatTensor(loss_ratio)

        for pre_beh in self.behaviors[:-1]:
            tgt_items = dict2set(self.train_user_behavior_dict[self.behaviors[-1]])
            pre_items = dict2set(self.train_user_behavior_dict[pre_beh])
            if self.loss_ratio_type == 'post_cond':
                denominator = tgt_items
            elif self.loss_ratio_type == 'jaccard':
                denominator = tgt_items | pre_items
            elif self.loss_ratio_type == 'pre_cond':
                denominator = pre_items
            else:
                raise NotImplementedError
            numerator = tgt_items & pre_items
            global_loss_ratios[pre_beh] = len(numerator) / len(denominator)
        logger.info('Generate {} loss ratios'.format(self.loss_ratio_type))

        return personal_loss_ratios, global_loss_ratios

    def get_interaction_mats(self):
        for behavior in self.behaviors:
            row = []
            col = []
            val = []
            for user, items in self.train_user_behavior_dict[behavior].items():
                for item in items:
                    row.append(int(user))
                    col.append(int(item))
                    val.append(1.0)
            mat = sp.coo_matrix((val, (row, col)), shape=(self.user_count + 1, self.item_count + 1))
            a = sp.csr_matrix((self.user_count + 1, self.user_count + 1))
            b = sp.csr_matrix((self.item_count + 1, self.item_count + 1))
            adj = sp.vstack([sp.hstack([a, mat]), sp.hstack([mat.transpose(), b])])
            self.behavior_mats[behavior] = mat
            self.behavior_adjs[behavior] = sp_mat_to_torch_sp_tensor(normalize_adj(adj.tocoo(), self.adj_type))

        logger.info('Generate {} adjacant matrices'.format(self.adj_type))

    def behavior_dataset(self):
        return MBTrainData(self.user_count, self.item_count, self.train_user_behavior_dict, self.behaviors)

    def validate_dataset(self):
        return MBTestData(self.user_count, self.item_count, samples=list(self.validation_dict.keys()))

    def test_dataset(self):
        return MBTestData(self.user_count, self.item_count, samples=list(self.test_dict.keys()))
