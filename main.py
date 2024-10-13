import argparse
import os
import time

import torch
from loguru import logger
from torch.utils.tensorboard import SummaryWriter

from dataset import DataSet
from model import DAGCN
from trainer import Trainer
from utils import set_seed

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Set args', add_help=False)

    # Data
    parser.add_argument('--data_name', type=str, default='taobao', help='choose in {tmall, taobao}')
    parser.add_argument('--model_name', type=str, default='DAGCN', help='model name')
    parser.add_argument('--model_path', type=str, default='./checkpoint', help='save model path')
    parser.add_argument('--data_path', type=str, default='./data', help='data path')
    parser.add_argument('--log_path', type=str, default='./log', help='log path')
    parser.add_argument('--gpu', type=int, default=1, help='gpu id')
    parser.add_argument('--seed', type=int, default=2023, help='seed')

    # Training & Evaluation
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--decay', type=float, default=0.0, help='decay')
    parser.add_argument('--epochs', type=str, default=1000, help='number of epochs')
    parser.add_argument('--early_stop', type=int, default=30, help='early stop threshold')

    parser.add_argument('--batch_size', type=int, default=4096, help='train batch size')
    parser.add_argument('--test_batch_size', type=int, default=8192, help='test batch size')
    parser.add_argument('--topk', type=list, default=[10, 20, 50, 80], help='top k')
    parser.add_argument('--metrics', type=list, default=['hit', 'ndcg'], help='metrics')

    # Model
    parser.add_argument('--embedding_size', type=int, default=64, help='embedding size')
    parser.add_argument('--if_load_model', action='store_true', help='load model')
    parser.add_argument('--checkpoint', type=str, default='', help='checkpoint path')

    # MTL
    parser.add_argument('--if_multi_tasks', action='store_true', help='multi-task learning')
    parser.add_argument('--mtl_type', type=str, default='personalized', help='choose in {global, personalized, none}')
    parser.add_argument('--loss_ratio_type', type=str, default='jaccard',
                        help='choose in {pre_cond, post_cond, jaccard}')
    parser.add_argument('--reg_weight', type=float, default=1e-3, help='regularization weight')
    parser.add_argument('--aux_weight', type=float, default=1.0, help='auxiliary weight')

    # Embedding & Behavior Conversion & Message Aggregation
    parser.add_argument('--keep_rate', type=float, default=1.0, help='keep rate')
    parser.add_argument('--if_layer_norm', action='store_true', help='layer norm')
    parser.add_argument('--if_add_weight', action='store_true', help='add weight')
    parser.add_argument('--if_add_bias', action='store_true', help='add bias')
    parser.add_argument('--adj_type', type=str, default='asym', help='choose in {asym, sym, mean, none}')
    parser.add_argument('--trans_type', type=str, default='post_cond', help='choose in {pre_cond, post_cond, jaccard}')
    parser.add_argument('--gcn_method', type=str, default='last', help='choose in {last, mean}')
    parser.add_argument('--behaviors', nargs='+', default=[], type=str, help='specific behaviors')

    args = parser.parse_args()

    # Seed and Device
    set_seed(args.seed)

    if torch.cuda.is_available() and args.gpu >= 0:
        args.device = torch.device('cuda:{}'.format(args.gpu))
    else:
        args.device = torch.device('cpu')

    # Log
    args.t = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime())
    full_log_path = os.path.join(args.log_path, args.t)
    if not os.path.exists(full_log_path):
        os.makedirs(full_log_path)
    logger.add(os.path.join(full_log_path, '{}_{}.log'.format(args.model_name, args.data_name)), encoding='utf-8')
    args.train_writer = SummaryWriter(os.path.join(full_log_path, '{}_{}_train'.format(args.model_name, args.data_name)))
    args.test_writer = SummaryWriter(os.path.join(full_log_path, '{}_{}_test'.format(args.model_name, args.data_name)))

    # Dataset and Model
    start = time.time()
    dataset = DataSet(args)
    model = DAGCN(args, dataset)
    model = model.to(args.device)

    # Train
    logger.info(model)
    trainer = Trainer(model, dataset, args)
    trainer.train_model()
    logger.info('total cost time %.2fs' % (time.time() - start))
    logger.info(args.__str__())
