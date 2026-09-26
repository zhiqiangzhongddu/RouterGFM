import torch as t
import torch.nn.functional as F
import utils.timelogger as logger
from utils.timelogger import log
from params import args
from model import AnyGraph, PredictionHead
from data_handler import MultiDataHandler
import numpy as np
import os
import time
import csv
import json
import sys
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from checkpoint_io import load_checkpoint_pair, save_checkpoint_pair


def _resolve_runtime_root():
    env_root = os.environ.get('ANYGRAPH_RUNTIME_ROOT', '').strip()
    if env_root != '':
        return env_root
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Exp:
    def __init__(self, multi_handler):
        self.multi_handler = multi_handler
        self.run_timestamp = int(time.time())
        self.base_dir = _resolve_runtime_root()
        self.history_dir = os.path.join(self.base_dir, 'History')
        self.models_dir = os.path.join(self.base_dir, 'Models')
        print(list(map(lambda x: x.data_name, multi_handler.trn_handlers)))
        for group_id, tst_handlers in enumerate(multi_handler.tst_handlers_group):
            print(f'Test group {group_id}', list(map(lambda x: x.data_name, tst_handlers)))
        self.metrics = dict()
        trn_mets = ['Loss', 'preLoss']
        tst_mets = ['Acc', 'F1']
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
        # Generic graph schema: one primary + one secondary metric per family
        # (single_label: acc/f1; multilabel: rocauc/ap; regression: mae/rmse).
        fieldnames = [
            'run_timestamp',
            'dataset_setting',
            'load_model',
            'save_path',
            'test_group_id',
            'dataset',
            'task_family',
            'repeat_times',
            'tst_num',
            'primary_metric',
            'primary_mean',
            'primary_std',
            'secondary_metric',
            'secondary_mean',
            'secondary_std',
        ]
        with open(csv_path, 'w', newline='') as fs:
            writer = csv.DictWriter(fs, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        log(f'Eval CSV Saved: {csv_path}')

    def save_oom_marker(self, oom_datasets):
        """Eval-only sidecar next to result_csv; the IcG parent (run.py) turns
        the listed datasets into result_status=OOM rows in the results table."""
        if not args.oom_tolerant or args.result_csv is None:
            return
        marker_path = args.result_csv + '.oom.json'
        payload = {'datasets': sorted(set(oom_datasets))}
        with open(marker_path, 'w', encoding='utf-8') as fs:
            json.dump(payload, fs)
        log(f'OOM marker saved: {marker_path} ({len(payload["datasets"])} dataset(s))')

    def run(self):
        self.prepare_model()
        log('Model Prepared')
        stloc = 0
        if args.load_model != None:
            self.load_model()
        # --- backbone training (link prediction over structural + star edges) ---
        for ep in range(stloc, args.epoch):
            start_time = time.time()
            self.model.assign_experts(self.multi_handler.trn_handlers, reca=True, log_assignment=True)
            reses = self.train_epoch()
            log(self.make_print('Train', ep, reses, False))
            self.multi_handler.remake_initial_projections()
            end_time = time.time()
            print(f'NOTICE: {end_time-start_time}')
            if (ep % args.tst_epoch == 0):
                self.save_history()
            print()

        # --- final evaluation, dispatched per dataset task_family ---
        eval_rows = []
        oom_datasets = list(getattr(self.multi_handler, 'oom_datasets', []))
        for test_group_id in range(len(self.multi_handler.tst_handlers_group)):
            tst_handlers = self.multi_handler.tst_handlers_group[test_group_id]
            if args.assignment == 'one-graph-one-expert':
                self.model.assign_experts(tst_handlers, reca=False, log_assignment=True)
            for i, handler in enumerate(tst_handlers):
                family = getattr(handler, 'task_family', 'single_label')
                try:
                    if family == 'single_label':
                        repeat_times = 10
                        accs, f1s = [], []
                        for _ in range(repeat_times):
                            handler.make_projectors()
                            if args.assignment != 'one-graph-one-expert':
                                self.model.assign_experts([handler], reca=False, log_assignment=False)
                            reses = self.test_epoch(handler, i if args.assignment == 'one-graph-one-expert' else 0)
                            accs.append(reses['Acc'])
                            f1s.append(reses['F1'])
                        accs, f1s = np.array(accs), np.array(f1s)
                        # NOTE: single_label uses argmax over class-nodes, so we report
                        # acc / macro-F1. ROC-AUC for binary sets (e.g. ogbg-molhiv) would
                        # require score output from pred_for_node_test (follow-up).
                        row = self._eval_row(
                            test_group_id, handler, family, repeat_times, int(reses['tstNum']),
                            'acc', accs.mean(), accs.std(),
                            'f1', f1s.mean(), f1s.std(),
                        )
                    else:
                        repeat_times = 5
                        if args.assignment != 'one-graph-one-expert':
                            self.model.assign_experts([handler], reca=False, log_assignment=False)
                        p1, p2, p_acc = [], [], []
                        n_eval = 0
                        for _ in range(repeat_times):
                            handler.make_projectors()
                            reses = self.eval_head_handler(handler, i if args.assignment == 'one-graph-one-expert' else 0)
                            p1.append(reses['primary'])
                            p2.append(reses['secondary'])
                            p_acc.append(reses.get('acc', 0.0))
                            n_eval = reses['tstNum']
                        p1, p2, p_acc = np.array(p1), np.array(p2), np.array(p_acc)
                        if family == 'multilabel':
                            # Report masked multi-label accuracy as the primary metric
                            # (matches the test_acc convention of the paper tables);
                            # keep ROC-AUC as the secondary metric.
                            row = self._eval_row(
                                test_group_id, handler, family, repeat_times, int(n_eval),
                                'acc', p_acc.mean(), p_acc.std(), 'rocauc', p1.mean(), p1.std(),
                            )
                        else:
                            row = self._eval_row(
                                test_group_id, handler, family, repeat_times, int(n_eval),
                                'mae', p1.mean(), p1.std(), 'rmse', p2.mean(), p2.std(),
                            )
                except t.cuda.OutOfMemoryError:
                    if not args.oom_tolerant:
                        raise
                    log(f'OOM while evaluating {handler.data_name}; skipping (oom_tolerant)')
                    oom_datasets.append(handler.data_name)
                    t.cuda.empty_cache()
                    continue
                eval_rows.append(row)
                log(f"[Eval] {handler.data_name} ({family}): "
                    f"{row['primary_metric']}={row['primary_mean']:.4f}±{row['primary_std']:.4f}, "
                    f"{row['secondary_metric']}={row['secondary_mean']:.4f}±{row['secondary_std']:.4f}")
        self.save_eval_csv(eval_rows)
        self.save_oom_marker(oom_datasets)
        self.save_history()

    def _eval_row(self, test_group_id, handler, family, repeat_times, tst_num,
                  m1, m1_mean, m1_std, m2, m2_mean, m2_std):
        return {
            'run_timestamp': self.run_timestamp,
            'dataset_setting': args.dataset_setting,
            'load_model': args.load_model if args.load_model is not None else '',
            'save_path': args.save_path,
            'test_group_id': test_group_id,
            'dataset': handler.data_name,
            'task_family': family,
            'repeat_times': repeat_times,
            'tst_num': int(tst_num),
            'primary_metric': m1,
            'primary_mean': float(m1_mean),
            'primary_std': float(m1_std),
            'secondary_metric': m2,
            'secondary_mean': float(m2_mean),
            'secondary_std': float(m2_std),
        }

    def eval_head_handler(self, handler, dataset_id):
        """Train a per-dataset MLP head on frozen super-node embeddings and
        evaluate it (multilabel: ROC-AUC/AP; regression: MAE/RMSE)."""
        self.model.eval()
        expert = self.model.summon(dataset_id)
        dev = args.devices[1]
        label_dim = int(handler.label_dim)
        labels = handler.labels.to(dev)            # (G, K)
        mask = handler.label_mask.to(dev)          # (G, K)
        off = int(handler.super_offset)

        def _rows(super_ids):
            # super-node global ids -> local graph rows in labels/mask
            return (super_ids.to(dev) - off).long()

        # Embed only the super-nodes we need (train+test), not all T nodes.
        # Exact for the default MLP backbone (node-wise); for a graph-transformer
        # backbone this is an approximation but keeps memory bounded on large
        # graph sets (pcba/qm9/...).
        n_train = int(handler.train_super.shape[0])
        pck = t.cat([handler.train_super, handler.test_super]).to(dev)
        with t.no_grad():
            emb_sub = expert.forward(handler.projectors, pck_nodes=pck).detach()  # (n_train+n_test, latdim)
        train_emb = emb_sub[:n_train]
        train_rows = _rows(handler.train_super)
        test_emb = emb_sub[n_train:]
        test_rows = _rows(handler.test_super)

        head = PredictionHead(emb_sub.shape[1], label_dim, hidden=args.head_hidden,
                              num_layers=args.head_layers, dropout=args.head_dropout).to(dev)
        opt = t.optim.Adam(head.parameters(), lr=args.head_lr, weight_decay=args.head_weight_decay)
        family = handler.task_family
        head.train()
        for _ in range(args.head_epoch):
            opt.zero_grad()
            pred = head(train_emb)
            y = labels[train_rows]
            m = mask[train_rows]
            if family == 'multilabel':
                loss = F.binary_cross_entropy_with_logits(pred, y, weight=m, reduction='sum') / (m.sum() + 1e-8)
            else:
                diff = (pred - y) * m
                loss = F.smooth_l1_loss(diff, t.zeros_like(diff), reduction='sum') / (m.sum() + 1e-8)
            loss.backward()
            opt.step()

        head.eval()
        with t.no_grad():
            pred = head(test_emb)
            y = labels[test_rows].cpu().numpy()
            m = mask[test_rows].cpu().numpy().astype(bool)
            if family == 'multilabel':
                prob = t.sigmoid(pred).cpu().numpy()
                aucs, aps = [], []
                for d in range(label_dim):
                    md = m[:, d]
                    if md.sum() == 0 or len(np.unique(y[md, d])) < 2:
                        continue
                    aucs.append(roc_auc_score(y[md, d], prob[md, d]))
                    aps.append(average_precision_score(y[md, d], prob[md, d]))
                primary = float(np.mean(aucs)) if aucs else 0.0
                secondary = float(np.mean(aps)) if aps else 0.0
                # Masked 0.5-threshold multi-label accuracy — comparable to the
                # test_acc reported for multi-label datasets elsewhere in the repo.
                acc = float(((prob >= 0.5) == (y >= 0.5))[m].mean()) if m.sum() else 0.0
            else:
                pred_np = pred.cpu().numpy()
                # de-standardize using stored train stats
                rmean = np.array(handler.reg_mean, dtype=np.float32) if handler.reg_mean else np.zeros(label_dim, dtype=np.float32)
                rstd = np.array(handler.reg_std, dtype=np.float32) if handler.reg_std else np.ones(label_dim, dtype=np.float32)
                if rmean.shape[0] != label_dim:
                    rmean = np.zeros(label_dim, dtype=np.float32)
                    rstd = np.ones(label_dim, dtype=np.float32)
                pred_de = pred_np * rstd[None, :] + rmean[None, :]
                y_de = y * rstd[None, :] + rmean[None, :]
                err = (pred_de - y_de)[m]
                primary = float(np.mean(np.abs(err))) if err.size else 0.0
                secondary = float(np.sqrt(np.mean(err ** 2))) if err.size else 0.0
                acc = 0.0
        t.cuda.empty_cache()
        return {'primary': primary, 'secondary': secondary, 'acc': acc,
                'tstNum': int(handler.test_super.shape[0])}

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

            expert = self.model.summon(dataset_id)#.cuda()
            opt = self.model.summon_opt(dataset_id)
            # adj = self.multi_handler.trn_handlers[dataset_id].trn_input_adj
            feats = self.multi_handler.trn_handlers[dataset_id].projectors
            loss, loss_dict = expert.cal_loss((ancs, poss, negs), feats)
            opt.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm_(expert.parameters(), max_norm=20, norm_type=2)
            opt.step()

            sample_num = ancs.shape[0]
            tot_samp_num += sample_num
            ep_loss += loss.item() * sample_num
            ep_preloss += loss_dict['preloss'].item() * sample_num
            ep_regloss += loss_dict['regloss'].item()
            log('Step %d/%d: loss = %.3f, pre = %.3f, reg = %.3f, pos = %.3f, neg = %.3f        ' % (i, steps, loss, loss_dict['preloss'], loss_dict['regloss'], loss_dict['posloss'], loss_dict['negloss']), save=False, oneline=True)

            counter[dataset_id] += 1
            if (counter[dataset_id] + 1) % self.multi_handler.trn_handlers[dataset_id].reproj_steps == 0:
            # if args.proj_trn_steps > 0 and counter[dataset_id] >= args.proj_trn_steps:
                self.multi_handler.trn_handlers[dataset_id].make_projectors()
            if (i + 1) % reassign_steps == 0:
                self.model.assign_experts(self.multi_handler.trn_handlers, reca=True, log_assignment=False)
        ret = dict()
        ret['Loss'] = ep_loss / tot_samp_num
        ret['preLoss'] = ep_preloss / tot_samp_num
        ret['regLoss'] = ep_regloss / steps
        t.cuda.empty_cache()
        return ret
    
    def make_trn_masks(self, numpy_usrs, csr_mat):
        trn_masks = csr_mat[numpy_usrs].tocoo()
        cand_size = trn_masks.shape[1]
        trn_masks = t.from_numpy(np.stack([trn_masks.row, trn_masks.col], axis=0)).long()
        return trn_masks, cand_size

    def test_loss_epoch(self, handler, dataset_id):
        with t.no_grad():
            tst_loader = handler.tst_loss_loader
            self.model.eval()
            expert = self.model.summon(dataset_id)#.cuda()
            ep_loss, ep_preloss, ep_regloss = 0, 0, 0
            steps = len(tst_loader)
            tot_samp_num = 0
            for i, batch_data in enumerate(tst_loader):
                ancs, poss, negs = batch_data
                ancs = ancs.long()
                poss = poss.long()
                negs = negs.long()
                # adj = handler.tst_input_adj
                feats = handler.projectors
                loss, loss_dict = expert.cal_loss((ancs, poss, negs), feats)
                
                sample_num = ancs.shape[0]
                tot_samp_num += sample_num
                ep_loss += loss.item() * sample_num
                ep_preloss += loss_dict['preloss'].item() * sample_num
                ep_regloss += loss_dict['regloss'].item()
                log('Step %d/%d: loss = %.3f, pre = %.3f, reg = %.3f, pos = %.3f, neg = %.3f        ' % (i, steps, loss, loss_dict['preloss'], loss_dict['regloss'], loss_dict['posloss'], loss_dict['negloss']), save=False, oneline=True)

        ret = dict()
        ret['Loss'] = ep_loss / tot_samp_num
        ret['preLoss'] = ep_preloss / tot_samp_num
        ret['regLoss'] = ep_regloss / steps
        ret['tot_samp_num'] = tot_samp_num
        t.cuda.empty_cache()
        return ret
    
    def test_epoch(self, handler, dataset_id):
        with t.no_grad():
            tst_loader = handler.tst_loader
            class_num = tst_loader.dataset.class_num
            self.model.eval()
            expert = self.model.summon(dataset_id)
            ep_acc, ep_tot = 0, 0
            all_preds, all_labels = None, None
            steps = len(tst_loader)
            for i, batch_data in enumerate(tst_loader):
                nodes, labels = list(map(lambda x: x.long().cuda(), batch_data))
                feats = handler.projectors
                preds = expert.pred_for_node_test(nodes, class_num, feats, rerun_embed=False if i!=0 else True)
                if i == 0:
                    all_preds, all_labels = preds, labels
                else:
                    all_preds = t.concatenate([all_preds, preds])
                    all_labels = t.concatenate([all_labels, labels])
                hit = (labels == preds).float().sum().item()
                ep_acc += hit
                ep_tot += labels.shape[0]
                log('Steps %d/%d: hit = %d, tot = %d          ' % (i, steps, ep_acc, ep_tot), save=False, oneline=True)
        ret = dict()
        # Guard against an empty test split (no graphs routed to test).
        ret['Acc'] = ep_acc / ep_tot if ep_tot > 0 else 0.0
        if all_preds is not None:
            ret['F1'] = f1_score(all_labels.cpu().numpy(), all_preds.cpu().numpy(), average='macro')
        else:
            ret['F1'] = 0.0
        ret['tstNum'] = ep_tot
        t.cuda.empty_cache()
        return ret

    
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
        # self.model.set_initial_projection(self.handler.torch_adj)
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
            'Use a built-in key (e.g., node) or comma-separated dataset names.'
        )

    datasets = dict()
    datasets['all'] = [
        'amazon-book', 'yelp2018', 'gowalla', 'yelp_textfeat', 'amazon_textfeat', 'steam_textfeat', 'Goodreads', 'Fitness', 'Photo', 'ml1m', 'ml10m', 'products_home', 'products_tech', 'cora', 'pubmed', 'citeseer', 'CS', 'arxiv', 'arxiv-ta', 'citation-2019', 'citation-classic', 'collab', 'ddi', 'ppa', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'email-Enron', 'web-Stanford', 'roadNet-PA', 'p2p-Gnutella06', 'soc-Epinions1',
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
    datasets['div1'] = [
        'products_tech', 'yelp2018', 'yelp_textfeat', 'products_home', 'steam_textfeat', 'amazon_textfeat', 'amazon-book', 'citation-2019', 'citation-classic', 'pubmed', 'citeseer', 'ppa', 'p2p-Gnutella06', 'soc-Epinions1', 'email-Enron',
    ]
    datasets['div2'] = [
        'Photo', 'Goodreads', 'Fitness', 'ml1m', 'ml10m', 'gowalla', 'arxiv', 'arxiv-ta', 'cora', 'CS', 'collab', 'proteins_spec0', 'proteins_spec1', 'proteins_spec2', 'proteins_spec3', 'ddi', 'web-Stanford', 'roadNet-PA',
    ]
    datasets['node'] = [
        'cora', 'arxiv', 'pubmed', 'home', 'tech'
    ]

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
            'Use built-in keys (e.g., node) or comma-separated dataset names.'
        )

    # trn_datasets = tst_datasets = ['products_home']
    if '+' not in args.dataset_setting:
        handler = MultiDataHandler(trn_datasets, [tst_datasets])
    else:
        handler = MultiDataHandler(trn_datasets, [trn_datasets, tst_datasets])
    log('Load Data')

    exp = Exp(handler)
    exp.run()
    print(args.load_model, args.dataset_setting)
