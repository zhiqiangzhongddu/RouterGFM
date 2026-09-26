import torch as t
import torch.nn.functional as F
import utils.timelogger as logger
from utils.timelogger import log
from params import args
from model import AnyGraph
from data_handler import MultiDataHandler
import numpy as np
import os
import time
import csv
from sklearn.metrics import roc_auc_score
from checkpoint_io import load_checkpoint_pair, save_checkpoint_pair


def _resolve_runtime_root():
    env_root = os.environ.get('ANYGRAPH_RUNTIME_ROOT', '').strip()
    if env_root != '':
        return env_root
    return os.path.dirname(os.path.abspath(__file__))


class Exp:
    def __init__(self, multi_handler):
        self.multi_handler = multi_handler
        self.run_timestamp = int(time.time())
        self.base_dir = _resolve_runtime_root()
        self.history_dir = os.path.join(self.base_dir, 'History')
        self.models_dir = os.path.join(self.base_dir, 'Models')
        self.eval_protocol = str(getattr(args, 'eval_protocol', 'agae')).lower()
        if self.eval_protocol not in {'agae', 'ranking'}:
            raise ValueError(f"Unsupported eval_protocol={self.eval_protocol}. Expected one of: agae, ranking.")
        print(list(map(lambda x: x.data_name, multi_handler.trn_handlers)))
        for group_id, tst_handlers in enumerate(multi_handler.tst_handlers_group):
            print(f'Test group {group_id}', list(map(lambda x: x.data_name, tst_handlers)))
        self.metrics = dict()
        trn_mets = ['Loss', 'preLoss']
        if self.eval_protocol == 'agae':
            tst_mets = ['ValAcc', 'Acc', 'Loss']
        else:
            tst_mets = ['Recall', 'NDCG', 'Loss', 'preLoss']
        mets = trn_mets + tst_mets
        for met in mets:
            if met in trn_mets:
                self.metrics['Train' + met] = list()
            if met in tst_mets:
                for i in range(len(self.multi_handler.tst_handlers_group)):
                    self.metrics['Test' + str(i) + met] = list()

    def make_print(self, name, ep, reses, save, data_name=None):
        if data_name is None:
            ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        else:
            ret = 'Epoch %d/%d, %s %s: ' % (ep, args.epoch, data_name, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric if data_name is None else name + data_name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '      '
        return ret

    def save_eval_csv(self, rows):
        if args.result_csv is None or len(rows) == 0:
            return
        csv_path = args.result_csv
        csv_dir = os.path.dirname(csv_path)
        if csv_dir != '':
            os.makedirs(csv_dir, exist_ok=True)
        fieldnames = [
            'run_timestamp',
            'dataset_setting',
            'load_model',
            'save_path',
            'test_group_id',
            'dataset',
            'eval_protocol',
            'topk',
            'repeat_times',
            'tst_num',
            'val_num',
            'recall_mean',
            'recall_std',
            'ndcg_mean',
            'ndcg_std',
            'acc_mean',
            'acc_std',
            'auc_mean',
            'auc_std',
            'loss_mean',
            'loss_std',
            'val_acc_mean',
            'val_acc_std',
            'val_auc_mean',
            'val_auc_std',
            'val_loss_mean',
            'val_loss_std',
            'threshold_mean',
            'threshold_std',
        ]
        with open(csv_path, 'w', newline='') as fs:
            writer = csv.DictWriter(fs, fieldnames=fieldnames, extrasaction='ignore')
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        log(f'Eval CSV Saved: {csv_path}')

    def run(self):
        self.prepare_model()
        log('Model Prepared')
        stloc = 0
        if args.load_model is not None:
            self.load_model()
            stloc = len(self.metrics['TrainLoss']) * args.tst_epoch - (args.tst_epoch - 1)
        best_metric = float('-inf')
        for ep in range(stloc, args.epoch):
            tst_flag = (ep % args.tst_epoch == 0)
            start_time = time.time()
            self.model.assign_experts(self.multi_handler.trn_handlers, reca=True, log_assignment=True)
            reses = self.train_epoch()
            log(self.make_print('Train', ep, reses, tst_flag))
            self.multi_handler.remake_initial_projections()
            end_time = time.time()
            print(f'NOTICE: {end_time-start_time}')
            if tst_flag:
                for handler_group_id in range(len(self.multi_handler.tst_handlers_group)):
                    tst_handlers = self.multi_handler.tst_handlers_group[handler_group_id]
                    self.model.assign_experts(tst_handlers, reca=False, log_assignment=True)
                    if self.eval_protocol == 'agae':
                        val_acc, tst_acc, tst_loss = 0.0, 0.0, 0.0
                        val_num, tst_num = 0, 0
                        for i, handler in enumerate(tst_handlers):
                            reses = self.test_epoch(handler, i)
                            val_acc += float(reses['ValAcc']) * int(reses['valNum'])
                            tst_acc += float(reses['Acc']) * int(reses['tstNum'])
                            tst_loss += float(reses['Loss']) * int(reses['tstNum'])
                            val_num += int(reses['valNum'])
                            tst_num += int(reses['tstNum'])
                        merged = {
                            'ValAcc': (val_acc / max(val_num, 1)),
                            'Acc': (tst_acc / max(tst_num, 1)),
                            'Loss': (tst_loss / max(tst_num, 1)),
                        }
                        log(self.make_print('Test' + str(handler_group_id), ep, merged, tst_flag))
                        if merged['ValAcc'] > best_metric:
                            best_metric = merged['ValAcc']
                    else:
                        recall, ndcg, tstnum = 0, 0, 0
                        for i, handler in enumerate(tst_handlers):
                            reses = self.test_epoch(handler, i)
                            recall += reses['Recall'] * reses['tstNum']
                            ndcg += reses['NDCG'] * reses['tstNum']
                            tstnum += reses['tstNum']
                        merged = {'Recall': recall / max(tstnum, 1), 'NDCG': ndcg / max(tstnum, 1)}
                        log(self.make_print('Test' + str(handler_group_id), ep, merged, tst_flag))
                        if merged['NDCG'] > best_metric:
                            best_metric = merged['NDCG']
                self.save_history()
            print()

        eval_rows = []
        for test_group_id in range(len(self.multi_handler.tst_handlers_group)):
            repeat_times = max(1, int(getattr(args, 'edge_eval_repeat_times', 5)))
            tst_handlers = self.multi_handler.tst_handlers_group[test_group_id]
            if self.eval_protocol == 'agae':
                overall_acc = np.zeros(repeat_times, dtype=np.float64)
                overall_auc = np.zeros(repeat_times, dtype=np.float64)
                overall_loss = np.zeros(repeat_times, dtype=np.float64)
                overall_tstnum = 0
                for handler in tst_handlers:
                    mets = dict()
                    for _ in range(repeat_times):
                        handler.make_projectors()
                        self.model.assign_experts([handler], reca=False, log_assignment=False)
                        reses = self.test_epoch(handler, 0)
                        for met in reses:
                            if met not in mets:
                                mets[met] = []
                            mets[met].append(reses[met])
                    tstnum = int(reses['tstNum'])
                    valnum = int(reses['valNum'])
                    tot_reses = dict()
                    for met in reses:
                        tem_arr = np.array(mets[met], dtype=np.float64)
                        tot_reses[met + '_std'] = tem_arr.std()
                        tot_reses[met + '_mean'] = tem_arr.mean()
                    overall_acc += np.array(mets['Acc'], dtype=np.float64) * tstnum
                    overall_auc += np.array(mets['AUC'], dtype=np.float64) * tstnum
                    overall_loss += np.array(mets['Loss'], dtype=np.float64) * tstnum
                    overall_tstnum += tstnum
                    eval_rows.append({
                        'run_timestamp': self.run_timestamp,
                        'dataset_setting': args.dataset_setting,
                        'load_model': args.load_model if args.load_model is not None else '',
                        'save_path': args.save_path,
                        'test_group_id': test_group_id,
                        'dataset': handler.data_name,
                        'eval_protocol': self.eval_protocol,
                        'repeat_times': repeat_times,
                        'tst_num': tstnum,
                        'val_num': valnum,
                        'acc_mean': float(tot_reses['Acc_mean']),
                        'acc_std': float(tot_reses['Acc_std']),
                        'auc_mean': float(tot_reses['AUC_mean']),
                        'auc_std': float(tot_reses['AUC_std']),
                        'loss_mean': float(tot_reses['Loss_mean']),
                        'loss_std': float(tot_reses['Loss_std']),
                        'val_acc_mean': float(tot_reses['ValAcc_mean']),
                        'val_acc_std': float(tot_reses['ValAcc_std']),
                        'val_auc_mean': float(tot_reses['ValAUC_mean']),
                        'val_auc_std': float(tot_reses['ValAUC_std']),
                        'val_loss_mean': float(tot_reses['ValLoss_mean']),
                        'val_loss_std': float(tot_reses['ValLoss_std']),
                        'threshold_mean': float(tot_reses['Threshold_mean']),
                        'threshold_std': float(tot_reses['Threshold_std']),
                    })
                    log(self.make_print('Test', args.epoch, {
                        'Acc_mean': float(tot_reses['Acc_mean']),
                        'Loss_mean': float(tot_reses['Loss_mean']),
                        'ValAcc_mean': float(tot_reses['ValAcc_mean']),
                    }, False, handler.data_name))
                if overall_tstnum > 0:
                    overall_acc /= overall_tstnum
                    overall_auc /= overall_tstnum
                    overall_loss /= overall_tstnum
                overall_res = dict()
                overall_res['Acc_mean'] = overall_acc.mean() if overall_tstnum > 0 else 0.0
                overall_res['Acc_std'] = overall_acc.std() if overall_tstnum > 0 else 0.0
                overall_res['AUC_mean'] = overall_auc.mean() if overall_tstnum > 0 else 0.0
                overall_res['AUC_std'] = overall_auc.std() if overall_tstnum > 0 else 0.0
                overall_res['Loss_mean'] = overall_loss.mean() if overall_tstnum > 0 else 0.0
                overall_res['Loss_std'] = overall_loss.std() if overall_tstnum > 0 else 0.0
                eval_rows.append({
                    'run_timestamp': self.run_timestamp,
                    'dataset_setting': args.dataset_setting,
                    'load_model': args.load_model if args.load_model is not None else '',
                    'save_path': args.save_path,
                    'test_group_id': test_group_id,
                    'dataset': '__OVERALL__',
                    'eval_protocol': self.eval_protocol,
                    'repeat_times': repeat_times,
                    'tst_num': int(overall_tstnum),
                    'acc_mean': float(overall_res['Acc_mean']),
                    'acc_std': float(overall_res['Acc_std']),
                    'auc_mean': float(overall_res['AUC_mean']),
                    'auc_std': float(overall_res['AUC_std']),
                    'loss_mean': float(overall_res['Loss_mean']),
                    'loss_std': float(overall_res['Loss_std']),
                })
                log(self.make_print('Overall Test', args.epoch, overall_res, False))
            else:
                overall_recall = np.zeros(repeat_times, dtype=np.float64)
                overall_ndcg = np.zeros(repeat_times, dtype=np.float64)
                overall_tstnum = 0
                for handler in tst_handlers:
                    for topk in [args.topk]:
                        args.topk = topk
                        mets = dict()
                        for _ in range(repeat_times):
                            handler.make_projectors()
                            self.model.assign_experts([handler], reca=False, log_assignment=False)
                            reses = self.test_epoch(handler, 0)
                            for met in reses:
                                if met not in mets:
                                    mets[met] = []
                                mets[met].append(reses[met])
                        tstnum = reses['tstNum']
                        tot_reses = dict()
                        for met in reses:
                            tem_arr = np.array(mets[met])
                            tot_reses[met + '_std'] = tem_arr.std()
                            tot_reses[met + '_mean'] = tem_arr.mean()
                        overall_recall += np.array(mets['Recall']) * tstnum
                        overall_ndcg += np.array(mets['NDCG']) * tstnum
                        overall_tstnum += tstnum
                        eval_rows.append({
                            'run_timestamp': self.run_timestamp,
                            'dataset_setting': args.dataset_setting,
                            'load_model': args.load_model if args.load_model is not None else '',
                            'save_path': args.save_path,
                            'test_group_id': test_group_id,
                            'dataset': handler.data_name,
                            'eval_protocol': self.eval_protocol,
                            'topk': topk,
                            'repeat_times': repeat_times,
                            'tst_num': int(tstnum),
                            'recall_mean': float(tot_reses['Recall_mean']),
                            'recall_std': float(tot_reses['Recall_std']),
                            'ndcg_mean': float(tot_reses['NDCG_mean']),
                            'ndcg_std': float(tot_reses['NDCG_std']),
                        })
                        log(self.make_print(f'Test Top-{topk}', args.epoch, tot_reses, False, handler.data_name))
                if overall_tstnum > 0:
                    overall_recall /= overall_tstnum
                    overall_ndcg /= overall_tstnum
                overall_res = dict()
                overall_res['Recall_mean'] = overall_recall.mean() if overall_tstnum > 0 else 0.0
                overall_res['Recall_std'] = overall_recall.std() if overall_tstnum > 0 else 0.0
                overall_res['NDCG_mean'] = overall_ndcg.mean() if overall_tstnum > 0 else 0.0
                overall_res['NDCG_std'] = overall_ndcg.std() if overall_tstnum > 0 else 0.0
                eval_rows.append({
                    'run_timestamp': self.run_timestamp,
                    'dataset_setting': args.dataset_setting,
                    'load_model': args.load_model if args.load_model is not None else '',
                    'save_path': args.save_path,
                    'test_group_id': test_group_id,
                    'dataset': '__OVERALL__',
                    'eval_protocol': self.eval_protocol,
                    'topk': args.topk,
                    'repeat_times': repeat_times,
                    'tst_num': int(overall_tstnum),
                    'recall_mean': float(overall_res['Recall_mean']),
                    'recall_std': float(overall_res['Recall_std']),
                    'ndcg_mean': float(overall_res['NDCG_mean']),
                    'ndcg_std': float(overall_res['NDCG_std']),
                })
                log(self.make_print('Overall Test', args.epoch, overall_res, False))
        self.save_eval_csv(eval_rows)
        self.save_history()

    def print_model_size(self):
        total_params = 0
        trainable_params = 0
        non_trainable_params = 0
        for param in self.model.parameters():
            tem = np.prod(param.size())
            total_params += tem
            if param.requires_grad:
                trainable_params += tem
            else:
                non_trainable_params += tem
        print(f'Total params: {total_params/1e6}')
        print(f'Trainable params: {trainable_params/1e6}')
        print(f'Non-trainable params: {non_trainable_params/1e6}')

    def prepare_model(self):
        self.model = AnyGraph()
        t.cuda.empty_cache()
        self.print_model_size()

    def train_epoch(self):
        self.model.train()
        trn_loader = self.multi_handler.joint_trn_loader
        trn_loader.dataset.neg_sampling()
        ep_loss, ep_preloss, ep_regloss = 0, 0, 0
        steps = len(trn_loader)
        tot_samp_num = 0
        counter = [0] * len(self.multi_handler.trn_handlers)
        reassign_steps = sum(list(map(lambda x: x.reproj_steps, self.multi_handler.trn_handlers)))
        for i, batch_data in enumerate(trn_loader):
            if args.epoch_max_step > 0 and i >= args.epoch_max_step:
                break
            ancs, poss, negs, dataset_id = batch_data
            ancs = ancs[0].long()
            poss = poss[0].long()
            negs = negs[0].long()
            dataset_id = dataset_id[0].long()
            tem_bar = self.multi_handler.trn_handlers[dataset_id].ratio_500_all
            if tem_bar < 1.0 and np.random.uniform() > tem_bar:
                steps -= 1
                continue

            expert = self.model.summon(dataset_id)
            opt = self.model.summon_opt(dataset_id)
            feats = self.multi_handler.trn_handlers[dataset_id].projectors
            loss, loss_dict = expert.cal_loss((ancs, poss, negs), feats)
            opt.zero_grad()
            loss.backward()
            opt.step()

            sample_num = ancs.shape[0]
            tot_samp_num += sample_num
            ep_loss += loss.item() * sample_num
            ep_preloss += loss_dict['preloss'].item() * sample_num
            ep_regloss += loss_dict['regloss'].item()
            log('Step %d/%d: loss = %.3f, pre = %.3f, reg = %.3f, pos = %.3f, neg = %.3f        ' % (i, steps, loss, loss_dict['preloss'], loss_dict['regloss'], loss_dict['posloss'], loss_dict['negloss']), save=False, oneline=True)

            counter[dataset_id] += 1
            if (counter[dataset_id] + 1) % self.multi_handler.trn_handlers[dataset_id].reproj_steps == 0:
                self.multi_handler.trn_handlers[dataset_id].make_projectors()
            if (i + 1) % reassign_steps == 0:
                self.model.assign_experts(self.multi_handler.trn_handlers, reca=True, log_assignment=False)
        ret = dict()
        ret['Loss'] = ep_loss / max(tot_samp_num, 1)
        ret['preLoss'] = ep_preloss / max(tot_samp_num, 1)
        ret['regLoss'] = ep_regloss / max(steps, 1)
        t.cuda.empty_cache()
        return ret

    def make_trn_masks(self, numpy_usrs, csr_mat):
        trn_masks = csr_mat[numpy_usrs].tocoo()
        cand_size = trn_masks.shape[1]
        trn_masks = t.from_numpy(np.stack([trn_masks.row, trn_masks.col], axis=0)).long()
        return trn_masks, cand_size

    def _get_final_embeds(self, expert, handler):
        try:
            final_embeds = expert.forward(handler.projectors)
        except Exception:
            final_embeds_list = []
            div = max(int(args.batch) * 3, 1)
            temlen = handler.projectors.shape[0] // div
            for i in range(temlen):
                st, ed = div * i, div * (i + 1)
                tem_projectors = handler.projectors[st:ed, :]
                final_embeds_list.append(expert.forward(tem_projectors))
            if temlen * div < handler.projectors.shape[0]:
                tem_projectors = handler.projectors[temlen * div :, :]
                final_embeds_list.append(expert.forward(tem_projectors))
            final_embeds = t.concat(final_embeds_list, dim=0)
        return final_embeds

    def _score_edge_pairs(self, expert, handler, edge_index, rerun_embed=True):
        if rerun_embed or not hasattr(expert, 'final_embeds'):
            expert.final_embeds = self._get_final_embeds(expert, handler)
        final_embeds = expert.final_embeds
        if edge_index.numel() == 0:
            return t.empty((0,), dtype=t.float32, device=final_embeds.device)
        edge_index = edge_index.to(args.devices[1])
        src, dst = edge_index[0], edge_index[1]
        return (final_embeds[src] * final_embeds[dst]).sum(-1)

    def _best_threshold_from_val(self, logits, labels):
        scores = logits.detach().cpu().numpy().astype(np.float64, copy=False)
        labs = labels.detach().cpu().numpy().astype(np.int64, copy=False)
        if scores.size == 0:
            return 0.0
        order = np.argsort(-scores, kind='mergesort')
        scores = scores[order]
        labs = labs[order]
        total = int(scores.shape[0])
        total_pos = int(labs.sum())
        total_neg = int(total - total_pos)

        best_acc = float(total_neg) / float(total)
        best_threshold = float(scores[0] + 1e-12)
        tp = 0
        fp = 0
        idx = 0
        while idx < total:
            score_val = scores[idx]
            grp_cnt = 0
            grp_pos = 0
            while idx < total and scores[idx] == score_val:
                grp_cnt += 1
                grp_pos += int(labs[idx])
                idx += 1
            tp += grp_pos
            fp += grp_cnt - grp_pos
            tn = total_neg - fp
            acc = float(tp + tn) / float(total)
            if acc > best_acc:
                best_acc = acc
                best_threshold = float(score_val)
        return best_threshold

    def _binary_metrics(self, logits, labels, threshold):
        if labels.numel() == 0:
            return 0.0, 0.0
        pred = (logits >= float(threshold)).to(labels.dtype)
        acc = (pred == labels).float().mean().item()
        loss = F.binary_cross_entropy_with_logits(logits, labels.float()).item()
        return acc, loss

    def _binary_auc(self, logits, labels):
        labs = labels.detach().cpu().numpy().astype(np.int64, copy=False)
        scores = logits.detach().cpu().numpy().astype(np.float64, copy=False)
        if labs.size == 0 or np.unique(labs).size < 2:
            return 0.5
        return float(roc_auc_score(labs, scores))

    def test_epoch_ranking(self, handler, dataset_id):
        with t.no_grad():
            tst_loader = handler.tst_loader
            self.model.eval()
            expert = self.model.summon(dataset_id)
            ep_recall, ep_ndcg = 0, 0
            ep_tstnum = len(tst_loader.dataset)
            steps = max(ep_tstnum // args.tst_batch, 1)
            for i, batch_data in enumerate(tst_loader):
                if args.tst_steps != -1 and i > args.tst_steps:
                    break
                usrs = batch_data.long()
                trn_masks, cand_size = self.make_trn_masks(batch_data.numpy(), tst_loader.dataset.csrmat)
                feats = handler.projectors
                all_preds = expert.pred_for_test((usrs, trn_masks), cand_size, feats, rerun_embed=False if i != 0 else True)
                _, top_locs = t.topk(all_preds, args.topk)
                top_locs = top_locs.cpu().numpy()
                recall, ndcg = self.calc_recall_ndcg(top_locs, tst_loader.dataset.tstLocs, usrs)
                ep_recall += recall
                ep_ndcg += ndcg
                log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)
        ret = dict()
        if args.tst_steps != -1:
            ep_tstnum = args.tst_steps * args.tst_batch
        ret['Recall'] = ep_recall / max(ep_tstnum, 1)
        ret['NDCG'] = ep_ndcg / max(ep_tstnum, 1)
        ret['tstNum'] = ep_tstnum
        t.cuda.empty_cache()
        return ret

    def test_epoch_agae(self, handler, dataset_id):
        with t.no_grad():
            payload = getattr(handler, 'edge_eval_payload', None)
            if payload is None:
                raise FileNotFoundError(
                    f'Edge eval payload not found for dataset={handler.data_name}. '
                    f'Expected file name: {args.edge_eval_payload_name}'
                )
            self.model.eval()
            expert = self.model.summon(dataset_id)

            val_pos = payload['val_pos_edge_index']
            val_neg = payload['val_neg_edge_index']
            tst_pos = payload['test_pos_edge_index']
            tst_neg = payload['test_neg_edge_index']

            val_edges = t.cat([val_pos, val_neg], dim=1)
            val_labels = t.cat([
                t.ones(val_pos.size(1), dtype=t.float32),
                t.zeros(val_neg.size(1), dtype=t.float32),
            ], dim=0).to(args.devices[1])

            tst_edges = t.cat([tst_pos, tst_neg], dim=1)
            tst_labels = t.cat([
                t.ones(tst_pos.size(1), dtype=t.float32),
                t.zeros(tst_neg.size(1), dtype=t.float32),
            ], dim=0).to(args.devices[1])

            val_logits = self._score_edge_pairs(expert, handler, val_edges, rerun_embed=True)
            tst_logits = self._score_edge_pairs(expert, handler, tst_edges, rerun_embed=False)

            threshold_mode = str(getattr(args, 'edge_eval_threshold_mode', 'val_best_acc')).lower()
            if threshold_mode == 'zero':
                threshold = 0.0
            else:
                threshold = self._best_threshold_from_val(val_logits, val_labels)

            val_acc, val_loss = self._binary_metrics(val_logits, val_labels, threshold)
            tst_acc, tst_loss = self._binary_metrics(tst_logits, tst_labels, threshold)
            val_auc = self._binary_auc(val_logits, val_labels)
            tst_auc = self._binary_auc(tst_logits, tst_labels)
            ret = {
                'ValAcc': float(val_acc),
                'ValLoss': float(val_loss),
                'ValAUC': float(val_auc),
                'Acc': float(tst_acc),
                'Loss': float(tst_loss),
                'AUC': float(tst_auc),
                'Threshold': float(threshold),
                'valNum': int(val_labels.numel()),
                'tstNum': int(tst_labels.numel()),
            }
            t.cuda.empty_cache()
            return ret

    def test_epoch(self, handler, dataset_id):
        if self.eval_protocol == 'agae':
            return self.test_epoch_agae(handler, dataset_id)
        return self.test_epoch_ranking(handler, dataset_id)

    def calc_recall_ndcg(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def save_history(self):
        if args.epoch == 0:
            return
        os.makedirs(self.history_dir, exist_ok=True)
        os.makedirs(self.models_dir, exist_ok=True)
        content = {
            'model': self.model,
        }
        save_checkpoint_pair(
            torch_module=t,
            model_path=os.path.join(self.models_dir, args.save_path + '.mod'),
            history_path=os.path.join(self.history_dir, args.save_path + '.his'),
            model_payload=content,
            history_payload=self.metrics,
        )
        log('Model Saved: %s' % args.save_path)

    def load_model(self):
        ckp, self.metrics = load_checkpoint_pair(
            torch_module=t,
            model_path=os.path.join(self.models_dir, args.load_model + '.mod'),
            history_path=os.path.join(self.history_dir, args.load_model + '.his'),
        )
        self.model = ckp['model']
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=0)
        log('Model Loaded')

if __name__ == '__main__':
    if getattr(args, 'seed', None) is not None:
        import random as _random
        _random.seed(int(args.seed))
        np.random.seed(int(args.seed))
        t.manual_seed(int(args.seed))
        if t.cuda.is_available():
            t.cuda.manual_seed_all(int(args.seed))
        log('Seeded RNGs with seed=%d' % int(args.seed))
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    if len(args.gpu.split(',')) == 2:
        args.devices = ['cuda:0', 'cuda:1']
    elif len(args.gpu.split(',')) > 2:
        raise Exception('Devices should be less than 2')
    else:
        args.devices = ['cuda:0', 'cuda:0']
    logger.saveDefault = True
    log('Start')

    if args.dataset_setting is None or str(args.dataset_setting).strip() == '' or args.dataset_setting == 'training':
        raise ValueError(
            'dataset_setting must be explicitly provided in strict baseline mode. '
            'Use a built-in key (e.g., link1) or comma-separated dataset names.'
        )

    datasets = dict()
    datasets['all'] = [
        'amazon-book', 'yelp2018', 'gowalla', 'yelp_textfeat', 'amazon_textfeat', 'steam_textfeat', 'Goodreads', 'Fitness', 'Photo', 'ml1m', 'ml10m', 'products_home', 'products_tech', 'cora', 'pubmed', 'citeseer', 'CS', 'arxiv', 'arxiv-ta', 'citation-2019', 'citation-classic', 'collab', 'ddi', 'ppa', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'email-Enron', 'web-Stanford', 'roadNet-PA', 'p2p-Gnutella06', 'soc-Epinions1'
    ]
    datasets['ecommerce'] = [
        'amazon-book', 'yelp2018', 'gowalla', 'yelp_textfeat', 'amazon_textfeat', 'steam_textfeat', 'Goodreads', 'Fitness', 'Photo', 'ml1m', 'ml10m', 'products_home', 'products_tech'
    ]
    datasets['academic'] = [
        'cora', 'pubmed', 'citeseer', 'CS', 'arxiv', 'arxiv-ta', 'citation-2019', 'citation-classic', 'collab'
    ]
    datasets['others'] = [
        'ddi', 'ppa', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'email-Enron', 'web-Stanford', 'roadNet-PA', 'p2p-Gnutella06', 'soc-Epinions1'
    ]
    datasets['link1'] = [
        'products_tech', 'yelp2018', 'yelp_textfeat', 'products_home', 'steam_textfeat', 'amazon_textfeat', 'amazon-book', 'citation-2019', 'citation-classic', 'pubmed', 'citeseer', 'ppa', 'p2p-Gnutella06', 'soc-Epinions1', 'email-Enron',
    ]
    datasets['link2'] = [
        'Photo', 'Goodreads', 'Fitness', 'ml1m', 'ml10m', 'gowalla', 'arxiv', 'arxiv-ta', 'cora', 'CS', 'collab', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'ddi', 'web-Stanford', 'roadNet-PA',
    ]
    excluded_excl8 = {
        'amazon-book', 'amazon_textfeat', 'gowalla', 'ml10m',
        'ml1m', 'steam_textfeat', 'yelp2018', 'yelp_textfeat',
    }
    datasets['link1_excl8'] = [d for d in datasets['link1'] if d not in excluded_excl8]
    datasets['link2_excl8'] = [d for d in datasets['link2'] if d not in excluded_excl8]

    def _resolve_dataset_token(token):
        token = token.strip()
        if token in datasets:
            return list(datasets[token])
        if token == '':
            return []
        return [x.strip() for x in token.split(',') if x.strip()]

    if args.dataset_setting in datasets.keys():
        trn_datasets = tst_datasets = list(datasets[args.dataset_setting])
    elif args.dataset_setting in datasets['all']:
        trn_datasets = tst_datasets = [args.dataset_setting]
    elif '+' in args.dataset_setting:
        idx = args.dataset_setting.index('+')
        trn_datasets = _resolve_dataset_token(args.dataset_setting[:idx])
        tst_datasets = _resolve_dataset_token(args.dataset_setting[idx+1:])
    elif '_in_' in args.dataset_setting:
        idx = args.dataset_setting.index('_in_')
        tst_datasets_1 = _resolve_dataset_token(args.dataset_setting[:idx])
        tst_datasets_2 = _resolve_dataset_token(args.dataset_setting[idx+len('_in_'):])
        tst_datasets = []
        for data in tst_datasets_1:
            if data in tst_datasets_2:
                tst_datasets.append(data)
        trn_datasets = tst_datasets
    else:
        trn_datasets = tst_datasets = _resolve_dataset_token(args.dataset_setting)

    if len(trn_datasets) == 0 or len(tst_datasets) == 0:
        raise ValueError(
            f'Failed to resolve datasets from dataset_setting="{args.dataset_setting}". '
            'Use built-in keys (e.g., link1, link2) or comma-separated dataset names.'
        )

    if '+' not in args.dataset_setting:
        # No zero-shot prediction test
        handler = MultiDataHandler(trn_datasets, [tst_datasets])
    else:
        handler = MultiDataHandler(trn_datasets, [trn_datasets, tst_datasets])
    log('Load Data')

    exp = Exp(handler)
    exp.run()
    print(args.load_model, args.dataset_setting)
