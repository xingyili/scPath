from scpath.models import MPNNs, DomainAdaptFusionModel, ReconstructionLoss
import os
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from tqdm import tqdm
from scpath.utils.data_prepare import prepare_graph_data
import torch.nn.functional as F
import numpy as np
from torch import optim
from easydl import inverseDecaySheduler, OptimWithSheduler, TrainingModeManager, OptimizerManager, AccuracyCounter
from easydl import one_hot, clear_output, variable_to_numpy
from scpath.utils.model_utils import MemoryQueue, GraphIDDataset
from scpath.utils.lib import sinkhorn, ubot_CCD, adaptive_filling, collate_fn, seed_everything
from scpath.evaluate import custom_eval
import ot
from tensorboardX import SummaryWriter
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse import find
from itertools import cycle
from collections import defaultdict
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.utils import from_scipy_sparse_matrix
from torch.utils.data import DataLoader, BatchSampler, SequentialSampler, RandomSampler
import time

def train_joint_with_da(adata_train, adata_test, knn_graphs, label_mapping, config, pathway_names):
    seed_everything(config.seed)
    assert adata_train.X.shape[1] == adata_test.X.shape[1], 'Reference and query feature dimensions do not match'
    for p in pathway_names:
        print(knn_graphs[f'train_{p}'].shape[0], len(adata_train))
        print(knn_graphs[f'test_{p}'].shape[0], len(adata_test))
        assert knn_graphs[f'train_{p}'].shape[0] == len(adata_train), f'{p}reference graph data are misaligned'
        assert knn_graphs[f'test_{p}'].shape[0] == len(adata_test), f'{p}query graph data are misaligned'
    class_counts = np.bincount(adata_train.obs['labels'])
    class_weights = 1.0 / (class_counts + 1e-06)
    class_weights = torch.FloatTensor(class_weights / class_weights.sum()).to(config.device)
    print(label_mapping)
    source_labels = adata_train.obs['labels'].unique()
    target_labels = adata_test.obs['labels'].unique()
    adata_train.obsm['X_emb'] = np.zeros((len(adata_train), config.fusion_hidden_dim))
    adata_train.obs['pred_labels'] = -1 * np.ones(len(adata_train), dtype=int)
    adata_test.obsm['X_emb'] = np.zeros((len(adata_test), config.fusion_hidden_dim))
    adata_test.obs['pred_labels'] = -1 * np.ones(len(adata_test), dtype=int)
    classes_set = {'source_classes': sorted(list(set(source_labels))), 'target_private_classes': sorted(list(set(target_labels) - set(source_labels))), 'common_classes': sorted(list(set(source_labels) & set(target_labels))), 'all_classes': sorted(list(set(source_labels) | set(target_labels)))}
    num_classes = len(classes_set['source_classes'])
    print(classes_set)
    base_models = {p: MPNNs(in_channels=config.input_dim, hidden_channels=config.base_hidden_dim, out_channels=num_classes, local_layers=config.mpnn_local_layers, gnn=config.gnn_type, res=True, bn=True, ln=True, pre_linear=True, pre_ln=True).to(config.device) for p in pathway_names}
    model = DomainAdaptFusionModel(base_models, num_classes, config, class_weights=class_weights).to(config.device)
    recon_loss_fn = ReconstructionLoss(pathways=pathway_names, base_models=base_models, loss_type='cosine', alpha=1, beta=0, gamma=0)

    def prepare_full_data(adata, graphs, data_type):
        return {p: prepare_graph_data(adata, graphs[f'{data_type}_{p}'], config.device) for p in pathway_names}
    source_inputs = prepare_full_data(adata_train, knn_graphs, 'train')
    target_inputs = prepare_full_data(adata_test, knn_graphs, 'test')
    label_t = np.asarray(list(adata_test.obs['labels']))
    feature_extractor = nn.ModuleList([model.base_models, model.feature_transform, model.cross_attn, model.gate, model.bottleneck]).to(config.device)
    source_train_ds = GraphIDDataset(adata_train, knn_graphs, 'train', pathway_names)
    target_train_ds = GraphIDDataset(adata_test, knn_graphs, 'test', pathway_names)
    source_test_ds = GraphIDDataset(adata_train, knn_graphs, 'train', pathway_names)
    target_test_ds = GraphIDDataset(adata_test, knn_graphs, 'test', pathway_names)
    num_workers = 4
    source_train_dl = DataLoader(source_train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    target_train_dl = DataLoader(target_train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    target_test_dl = DataLoader(target_test_ds, batch_size=config.eval_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    target_initMQ_dl = DataLoader(target_train_ds, batch_size=config.batch_size, shuffle=True, drop_last=True, collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    source_test_dl = DataLoader(dataset=source_test_ds, batch_size=config.eval_batch_size, shuffle=False, drop_last=False, collate_fn=collate_fn)
    classifier = model.classifier.to(config.device)
    cluster_head = model.cluster_head.to(config.device)
    optim_featex = optim.SGD(list(feature_extractor.parameters()), lr=config.fusion_lr, momentum=0.9)
    optim_cls = optim.SGD(classifier.parameters(), lr=config.fusion_lr, momentum=0.9)
    optim_cluhead = optim.SGD(cluster_head.parameters(), lr=config.fusion_lr, momentum=0.9)
    scheduler = lambda step, initial_lr: inverseDecaySheduler(step, initial_lr, gamma=10, power=0.75, max_iter=config.max_steps)
    opt_sche_featex = OptimWithSheduler(optim_featex, scheduler)
    opt_sche_cls = OptimWithSheduler(optim_cls, scheduler)
    opt_sche_cluhead = OptimWithSheduler(optim_cluhead, scheduler)
    if config.enhance_integration:
        domain_disc = model.domain_discriminator.to(config.device)
        optim_disc = optim.SGD(domain_disc.parameters(), lr=config.fusion_lr * 0.1, momentum=0.9)
        opt_sche_disc = OptimWithSheduler(optim_disc, scheduler)
    model.train()
    n_batch = config.memory_queue_size // config.batch_size
    memqueue = MemoryQueue(config.fusion_hidden_dim, config.batch_size, n_batch, config.device, config.temp).to(config.device)
    cnt_i = 0
    with TrainingModeManager([model], train=False), torch.no_grad():
        for target_batch in target_initMQ_dl:
            (target_graphs_dict, target_ids) = target_batch
            target_inputs = {p: (target_graphs_dict[p]['x'].to(config.device), target_graphs_dict[p]['edge_index'].to(config.device)) for p in pathway_names}
            target_outputs = model(target_inputs, mode='target')
            memqueue.update_queue(target_outputs['norm_feat'], target_ids.to(config.device))
            cnt_i += 1
            if cnt_i >= n_batch:
                break
    total_steps = tqdm(range(config.max_steps), desc='global step')
    logger = SummaryWriter(config.log_dir)
    beta = None
    best_results = None
    cls_labels = torch.LongTensor(adata_train.obs['labels'].values).to(config.device)
    global_step = 0
    source_batches = len(source_train_dl)
    target_batches = len(target_train_dl)
    max_iterations = max(source_batches, target_batches)
    while global_step < config.max_steps:
        for (i, (source_batch, target_batch)) in enumerate(zip(cycle(source_train_dl), cycle(target_train_dl))):
            if i >= max_iterations:
                break
            (source_pathway_data, source_ids) = source_batch
            (target_pathway_data, target_ids) = target_batch
            source_inputs = {p: (source_pathway_data[p]['x'].to(config.device), source_pathway_data[p]['edge_index'].to(config.device)) for p in pathway_names}
            target_inputs = {p: (target_pathway_data[p]['x'].to(config.device), target_pathway_data[p]['edge_index'].to(config.device)) for p in pathway_names}
            source_outputs = model(source_inputs, mode='source', original_inputs=source_inputs, class_labels=cls_labels[source_ids])
            target_outputs = model(target_inputs, mode='target', original_inputs=target_inputs)
            if 'reconstructions' in source_outputs:
                (_, recon_loss_source) = recon_loss_fn(source_outputs, source_inputs)
            else:
                recon_loss_source = 0.0
            if 'reconstructions' in target_outputs:
                (_, recon_loss_target) = recon_loss_fn(target_outputs, target_inputs)
            else:
                recon_loss_target = 0.0
            assert feature_extractor.training, 'Feature extractor must be in training mode'
            assert classifier.training, 'Classifier must be in training mode'
            assert (cls_labels >= 0).all() and (cls_labels < num_classes).all(), 'Label out of range!'
            batch_labels = cls_labels[source_ids]
            class_counts = torch.bincount(batch_labels, minlength=num_classes).float()
            class_weights = 1.0 / (class_counts + 1e-6)
            class_weights = class_weights / class_weights.sum()
            weights_factor = torch.clamp(class_weights * num_classes * config.class_weight_factor, min=0.1, max=10.0)
            loss_cls = F.cross_entropy(source_outputs['cls_output'], batch_labels, weight=weights_factor)
            norm_feat_t = target_outputs['norm_feat']
            after_cluhead_t = target_outputs['after_cluhead']
            minibatch_size = norm_feat_t.size(0)
            feat_mat2 = torch.matmul(norm_feat_t, norm_feat_t.t()) / config.temp
            mask = torch.eye(feat_mat2.size(0), feat_mat2.size(0)).bool().to(config.device)
            feat_mat2.masked_fill_(mask, -1 / config.temp)
            (nb_value_tt, nb_feat_tt) = memqueue.get_nearest_neighbor(norm_feat_t, target_ids.to(config.device))
            neighbor_candidate_sim = torch.cat([nb_value_tt.reshape(-1, 1), feat_mat2], 1)
            (values, indices) = torch.max(neighbor_candidate_sim, 1)
            valid_mask = values > 0
            indices = torch.where(valid_mask, indices, torch.zeros_like(indices))
            neighbor_norm_feat = torch.zeros((minibatch_size, norm_feat_t.shape[1])).to(config.device)
            for i in range(minibatch_size):
                neighbor_candidate_feat = torch.cat([nb_feat_tt[i].reshape(1, -1), norm_feat_t], 0)
                neighbor_norm_feat[i, :] = neighbor_candidate_feat[indices[i], :]
            neighbor_output = model.cluster_head(neighbor_norm_feat)
            fill_size_ot = config.batch_size
            mqfill_feat_t = memqueue.random_sample(fill_size_ot)
            mqfill_output_t = model.cluster_head(mqfill_feat_t)
            S_tt = torch.cat([after_cluhead_t, neighbor_output, mqfill_output_t], 0)
            S_tt *= config.temp
            Q_tt = sinkhorn(S_tt.detach(), epsilon=0.05, sinkhorn_iterations=7, global_step=global_step)
            Q_tt_tilde = Q_tt * Q_tt.size(0)
            anchor_Q = Q_tt_tilde[:minibatch_size, :]
            neighbor_Q = Q_tt_tilde[minibatch_size:2 * minibatch_size, :]
            loss_local = 0
            for i in range(minibatch_size):
                sub_loss_local = 0
                sub_loss_local += -torch.sum(neighbor_Q[i, :] * F.log_softmax(after_cluhead_t[i, :]))
                sub_loss_local += -torch.sum(anchor_Q[i, :] * F.log_softmax(neighbor_output[i, :]))
                sub_loss_local /= 2
                loss_local += sub_loss_local
            loss_local /= minibatch_size
            loss_global = -torch.mean(torch.sum(anchor_Q * F.log_softmax(after_cluhead_t, dim=1), dim=1))
            loss_pcd = (loss_global + loss_local) / 2
            loss_ccd = torch.tensor(0.0).to(config.device)
            if global_step > config.warmup_steps:
                source_prototype = model.classifier.fc.weight.data
                if beta is None:
                    beta = ot.unif(source_prototype.size(0))
                fill_size_uot = n_batch * config.batch_size
                mqfill_feat_t = memqueue.random_sample(fill_size_uot)
                ubot_feature_t = torch.cat([mqfill_feat_t, norm_feat_t], 0)
                full_size = ubot_feature_t.size(0)
                (newsim, fake_size) = adaptive_filling(ubot_feature_t, source_prototype, config.device, config.gamma, beta, fill_size_uot, high=config.high)
                (high_conf_label_id, high_conf_label, _, new_beta, _) = ubot_CCD(newsim, beta, config.device, fake_size=fake_size, fill_size=fill_size_uot, mode='minibatch', high=config.high)
                beta = config.mu * beta + (1 - config.mu) * new_beta
                if high_conf_label_id.size(0) > 0:
                    logits_t = target_outputs['cls_output'][high_conf_label_id]
                    loss_ccd = F.cross_entropy(logits_t, high_conf_label[high_conf_label_id])
                else:
                    loss_ccd = 0
            loss_adv = torch.tensor(0.0).to(config.device)
            if config.enhance_integration:
                p = float(global_step) / config.max_steps
                alpha = 2.0 / (1.0 + np.exp(-10 * p)) - 1
                d_out_s = model.domain_discriminator(source_outputs['norm_feat'], alpha)
                d_out_t = model.domain_discriminator(target_outputs['norm_feat'], alpha)
                loss_adv_s = F.binary_cross_entropy(d_out_s, torch.zeros_like(d_out_s))
                loss_adv_t = F.binary_cross_entropy(d_out_t, torch.ones_like(d_out_t))
                loss_adv = (loss_adv_s + loss_adv_t) / 2
            total_loss = config.lambda_cls * loss_cls + config.lambda_pcd * loss_pcd + config.lambda_ccd * loss_ccd + config.lambda_recon * (recon_loss_source + recon_loss_target)
            if config.enhance_integration:
                total_loss += config.lambda_adv * loss_adv
            optimizers_list = [opt_sche_featex, opt_sche_cls, opt_sche_cluhead]
            if config.enhance_integration:
                optimizers_list.append(opt_sche_disc)
            with OptimizerManager(optimizers_list):
                total_loss.backward()
            model.classifier.weight_norm()
            model.cluster_head.weight_norm()
            memqueue.update_queue(norm_feat_t.detach(), target_ids.to(config.device))
            global_step += 1
            total_steps.update()
            if global_step % config.log_interval == 0:
                counter = AccuracyCounter()
                counter.addOneBatch(variable_to_numpy(one_hot(cls_labels[source_ids], len(classes_set['source_classes']))), variable_to_numpy(source_outputs['cls_output']))
                acc_source = torch.tensor([counter.reportAccuracy()]).to(config.device)
                logger.add_scalar('loss_all', total_loss, global_step)
                logger.add_scalar('loss_cls', loss_cls, global_step)
                logger.add_scalar('loss_pcd', loss_pcd, global_step)
                logger.add_scalar('loss_ccd', loss_ccd, global_step)
                logger.add_scalar('acc_source', acc_source, global_step)
                if config.enhance_integration:
                    logger.add_scalar('loss_adv', loss_adv, global_step)
                if 'reconstructions' in source_outputs:
                    (recon_loss_dict, _) = recon_loss_fn(source_outputs, source_inputs)
                    for (loss_name, loss_val) in recon_loss_dict.items():
                        logger.add_scalar(f'recon_{loss_name}', loss_val, global_step)
            if global_step % config.test_interval == 0:
                (results, source_emb, source_pred, target_emb, target_pred, _) = custom_eval(model, config, target_test_dl, source_test_dl, classes_set, gamma=config.gamma, target_label=adata_test.obs['labels'].values, source_label=adata_train.obs['labels'].values, beta=beta, plot=False, log_dir=config.log_dir, epoch=global_step, high=config.high, label_mapping=label_mapping)
                if 'cls_common_acc' in results:
                    logger.add_scalar('cls_common_acc', results['cls_common_acc'], global_step)
                if 'cls_overall_acc' in results:
                    logger.add_scalar('cls_overall_acc', results['cls_overall_acc'], global_step)
                if 'cls_tp_acc' in results:
                    logger.add_scalar('cls_tp_acc', results['cls_tp_acc'], global_step)
                if 'h_score' in results:
                    logger.add_scalar('h_score', results['h_score'], global_step)
    print('Training completed; running final evaluation...')
    (final_results, source_emb, source_pred, target_emb, target_pred, save_pack) = custom_eval(model, config, target_test_dl, source_test_dl, classes_set, gamma=config.gamma, target_label=adata_test.obs['labels'].values, source_label=adata_train.obs['labels'].values, beta=beta, plot=False, log_dir=config.log_dir, high=config.high, portal=False, label_mapping=label_mapping)
    assert len(save_pack['ids']) == len(adata_test), 'Evaluation size does not match the query size'
    returned_ids = save_pack['ids']
    n_test = len(adata_test)
    full_ot_matrix = np.zeros((n_test, save_pack['ot_matrix'].shape[1]))
    full_cls_probs = np.zeros((n_test, save_pack['cls_probs'].shape[1]))
    full_emb = np.zeros((n_test, save_pack['embedding'].shape[1]))
    full_cls_preds = -1 * np.ones(n_test, dtype=int)
    full_ot_preds_common = -1 * np.ones(n_test, dtype=int)
    full_ot_preds_private = -1 * np.ones(n_test, dtype=int)
    full_dummy_mass = np.zeros(n_test)
    full_conf_scores = np.zeros(n_test)
    full_ot_matrix[returned_ids] = save_pack['ot_matrix']
    full_cls_probs[returned_ids] = save_pack['cls_probs']
    full_emb[returned_ids] = save_pack['embedding']
    full_cls_preds[returned_ids] = save_pack['cls_preds']
    full_ot_preds_common[returned_ids] = save_pack['ot_preds_common']
    full_ot_preds_private[returned_ids] = save_pack['ot_preds_private']
    full_dummy_mass[returned_ids] = save_pack['ot_dummy_mass']
    full_conf_scores[returned_ids] = save_pack['ot_conf_scores']
    adata_test.obsm['X_emb'] = full_emb
    adata_test.obsm['ot_matrix'] = full_ot_matrix
    adata_test.obsm['cls_probs'] = full_cls_probs
    assert 'ot_matrix' in adata_test.obsm, 'ot_matrix was not saved!'
    print(f"✓ ot_matrix shape: {adata_test.obsm['ot_matrix'].shape}")
    print(f"  Sample row sum: {adata_test.obsm['ot_matrix'][0].sum():.6f}")
    adata_test.obs['pred_labels_cls'] = full_cls_preds
    adata_test.obs['pred_labels_ot_common'] = full_ot_preds_common
    adata_test.obs['pred_labels_ot_private'] = full_ot_preds_private
    adata_test.obs['ot_dummy_mass'] = full_dummy_mass
    adata_test.obs['ot_conf_scores'] = full_conf_scores
    adata_test.obs['pred_labels'] = target_pred
    if len(source_emb) == len(adata_train):
        adata_train.obsm['X_emb'] = source_emb
        adata_train.obs['pred_labels'] = source_pred
    else:
        print('Warning: Source embedding size mismatch, skipping save for train set.')
    print(f'Saving results to {config.log_dir}')
    adata_train.write(os.path.join(config.log_dir, 'train_results.h5ad'))
    adata_test.write(os.path.join(config.log_dir, 'test_results.h5ad'))
    final_data = {'model': model, 'num_classes': num_classes, 'config': config.__dict__, 'beta': beta, 'final_results': final_results}
    torch.save(model, os.path.join(config.log_dir, 'model.pt'))
    with open(os.path.join(config.log_dir, 'final_model.pkl'), 'wb') as f:
        torch.save(final_data, f)
    final_state = {'fusion_model': model.state_dict(), 'base_models': {p: base_models[p].state_dict() for p in pathway_names}, 'config': config.__dict__, 'class_weights': class_weights, 'num_classes': num_classes, 'beta': beta, 'final_results': final_results, 'label_mapping': label_mapping}
    torch.save(final_state, os.path.join(config.log_dir, 'final_model.pt'))
    results_df = pd.DataFrame({'Final': pd.Series(final_results)})
    results_df.to_csv(os.path.join(config.log_dir, 'training_results.csv'))
