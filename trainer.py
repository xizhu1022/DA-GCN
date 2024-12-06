import os
import time

import numpy as np
import torch
from loguru import logger
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import DataSet
from metrics import metrics_dict


class Trainer(object):

    def __init__(self, model, dataset: DataSet, args):
        self.model = model.to(args.device)
        self.dataset = dataset
        self.behaviors = dataset.behaviors
        self.topk = args.topk
        self.metrics = args.metrics
        self.learning_rate = args.lr
        self.weight_decay = args.decay
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.epochs = args.epochs
        self.model_path = args.model_path
        self.model_name = args.model_name
        self.train_writer = args.train_writer
        self.test_writer = args.test_writer
        self.device = args.device
        self.t = args.t
        self.early_stop = args.early_stop

        self.optimizer = self.get_optimizer(self.model)

    def get_optimizer(self, model):
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=self.learning_rate,
                               weight_decay=self.weight_decay)
        return optimizer

    @logger.catch()
    def train_model(self):
        train_dataset_loader = DataLoader(dataset=self.dataset.behavior_dataset(),
                                          batch_size=self.batch_size,
                                          shuffle=True)

        best_valid_result = 0
        best_valid_dict = {}
        best_valid_epoch = 0
        final_test = None

        best_test_result = 0
        best_test_dict = {}
        best_test_epoch = 0
        for epoch in range(self.epochs):
            self.model.train()
            test_metric_dict, validate_metric_dict = self._train_one_epoch(train_dataset_loader, epoch)
            valid_result = validate_metric_dict['ndcg@10']
            test_result = test_metric_dict['hit@10']

            # update
            if valid_result - best_valid_result > 0:
                final_test = test_metric_dict
                best_valid_result = valid_result
                best_valid_dict = validate_metric_dict
                best_valid_epoch = epoch

            if test_result - best_test_result > 0:
                best_test_result = test_result
                best_test_dict = test_metric_dict
                best_test_epoch = epoch
                self.save_model(self.model)
                logger.info(f"model saved at epoch %d" % (epoch + 1))

            # early stop
            if epoch - best_valid_epoch > self.early_stop:
                break

        logger.info(f"training end, best valid epoch %d, results: %s" %
                    (best_valid_epoch + 1, best_valid_dict.__str__()))

        logger.info(f"final test result:  %s" % final_test.__str__())

        logger.info(f"best test epoch %d, results: %s" %
                    (best_test_epoch + 1, best_test_dict.__str__()))

    def _train_one_epoch(self, behavior_dataset_loader, epoch):
        start_time = time.time()
        behavior_dataset_iter = (
            tqdm(
                enumerate(behavior_dataset_loader),
                total=len(behavior_dataset_loader),
                desc=f"\033[1;35m Train {epoch + 1:>5}\033[0m"
            )
        )

        # train
        total_loss = 0.0
        batch_no = 0
        for batch_index, batch_data in behavior_dataset_iter:
            batch_data = batch_data.to(self.device)
            self.optimizer.zero_grad()
            loss = self.model(batch_data)
            loss.backward()
            self.optimizer.step()
            batch_no = batch_index + 1
            total_loss += loss.item()
        total_loss = total_loss / batch_no

        self.train_writer.add_scalar('total train loss', total_loss, epoch + 1)
        epoch_time = time.time() - start_time
        logger.info('epoch %d, time %.2fs, train loss: %.4f ' % (epoch + 1, epoch_time, total_loss))

        # validate
        start_time = time.time()
        validate_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.validate_dataset(),
                                             self.dataset.validation_dict, self.dataset.validation_gt_length,
                                             self.train_writer)
        epoch_time = time.time() - start_time
        logger.info(
            f"validation %d, time %.2fs, result: %s " % (epoch + 1, epoch_time, validate_metric_dict.__str__()))

        # test
        start_time = time.time()
        test_metric_dict = self.evaluate(epoch, self.test_batch_size, self.dataset.test_dataset(),
                                         self.dataset.test_dict, self.dataset.test_gt_length,
                                         self.test_writer)
        epoch_time = time.time() - start_time
        logger.info(f"test %d, time %.2fs, result: %s " % (epoch + 1, epoch_time, test_metric_dict.__str__()))

        return test_metric_dict, validate_metric_dict

    @logger.catch()
    @torch.no_grad()
    def evaluate(self, epoch, test_batch_size, dataset, gt_interacts, gt_length, writer):
        data_loader = DataLoader(dataset=dataset, batch_size=test_batch_size)

        self.model.eval()
        iter_data = (
            tqdm(
                enumerate(data_loader),
                total=len(data_loader),
                desc=f"\033[1;35mEvaluate \033[0m"
            )
        )
        topk_list = []
        train_items = self.dataset.train_user_behavior_dict[self.behaviors[-1]]
        for batch_index, batch_data in iter_data:
            batch_data = batch_data.to(self.device)
            scores = self.model.full_predict(batch_data)  # (test_bsz, |I|)

            batch_data = batch_data.to('cpu')
            scores = scores.to('cpu')
            for index, user in enumerate(batch_data):
                user_score = scores[index]
                items = train_items.get(user.item(), None)
                if items is not None:
                    user_score[items] = -np.inf
                _, topk_idx = torch.topk(user_score, max(self.topk), dim=-1)
                gt_items = gt_interacts[user.item()]
                mask = np.isin(topk_idx.to('cpu'), gt_items)
                topk_list.append(mask)

        topk_list = np.array(topk_list)
        metric_dict = self.calculate_result(topk_list, gt_length)
        for key, value in metric_dict.items():
            writer.add_scalar('evaluate ' + key, value, epoch + 1)
        return metric_dict

    def calculate_result(self, topk_list, gt_len):
        result_list = []
        for metric in self.metrics:
            metric_fuc = metrics_dict[metric.lower()]
            result = metric_fuc(topk_list, gt_len)
            result_list.append(result)
        result_list = np.stack(result_list, axis=0).mean(axis=1)
        metric_dict = {}
        for topk in self.topk:
            for metric, value in zip(self.metrics, result_list):
                key = '{}@{}'.format(metric, topk)
                metric_dict[key] = np.round(value[topk - 1], 4)

        return metric_dict

    def save_model(self, model):
        full_model_path = os.path.join(self.model_path, self.t)
        if not os.path.exists(full_model_path):
            os.makedirs(full_model_path)
        torch.save(model.state_dict(), os.path.join(full_model_path, 'model.pth'))
        torch.save(model.storage_all_embeddings, os.path.join(full_model_path, 'all_embeddings.pth'))
