import numpy as np
import torch.nn.functional as F
import pandas as pd
import torch.nn as nn
import torch
from scpath.utils.lib import ubot_CCD, adaptive_filling
from easydl import TrainingModeManager, Accumulator
import faiss
from tqdm import tqdm
import os
from sklearn.metrics import accuracy_score
from scpath.plot_res import plot_separate_figures, set_nature_style

def calculate_accuracy_simple(true_labels, pred_labels, mask=None):
    if mask is not None:
        return accuracy_score(true_labels[mask], pred_labels[mask])
    return accuracy_score(true_labels, pred_labels)

def save_predictions(id, true_labels, pred_labels, save_dir):
    df = pd.DataFrame({'id': id, 'true_label': true_labels, 'pred_label': pred_labels})
    save_path = os.path.join(save_dir, 'predictions.csv')
    df.to_csv(save_path, index=False)

def custom_eval(model, config, target_test_dl, source_test_dl, classes_set, gamma=0.7, target_label=None, source_label=None, beta=None, seed=None, uniformed_index=None, plot=False, log_dir=None, epoch=None, high=True, portal=False, label_mapping=None):
    if seed is None:
        seed = int(os.environ['PYTHONHASHSEED'])
    num_source_classes = len(classes_set['source_classes'])
    private_class_idx = num_source_classes
    if beta is None:
        beta = torch.ones(num_source_classes).to(model.device) / num_source_classes
    model.eval()
    with TrainingModeManager([model], train=False) as mgr, Accumulator(['label_t', 'norm_feat_t', 'cls_output_t', 'ids']) as eval_accumulator, torch.no_grad():
        for (target_graphs_dict, target_ids) in tqdm(target_test_dl, desc='Evaluating on target domain'):
            target_graphs_dict = {p: (data['x'].to(model.device), data['edge_index'].to(model.device)) for (p, data) in target_graphs_dict.items()}
            target_ids = target_ids.to(model.device)
            target_outputs = model(target_graphs_dict, mode='target')
            val = dict()
            val['norm_feat_t'] = target_outputs['norm_feat'].cpu().numpy()
            val['label_t'] = target_label[target_ids.cpu().numpy()]
            val['cls_output_t'] = target_outputs['cls_output'].cpu().numpy()
            val['ids'] = target_ids.cpu().numpy()
            eval_accumulator.updateData(val)
    for x in eval_accumulator:
        val[x] = eval_accumulator[x]
    norm_feat_t = val['norm_feat_t']
    label_t = val['label_t']
    cls_output_t = val['cls_output_t']
    ids_t = val['ids']
    del val
    with TrainingModeManager([model], train=False) as mgr, Accumulator(['label_s', 'norm_feat_s']) as eval_accumulator, torch.no_grad():
        for (source_graphs_dict, source_ids) in tqdm(source_test_dl, desc='Evaluating on source domain'):
            source_graphs_dict = {p: (data['x'].to(model.device), data['edge_index'].to(model.device)) for (p, data) in source_graphs_dict.items()}
            source_ids = source_ids.to(model.device)
            source_outputs = model(source_graphs_dict, mode='source')
            val = dict()
            val['norm_feat_s'] = source_outputs['norm_feat'].cpu().numpy()
            val['label_s'] = source_label[source_ids.cpu().numpy()]
            eval_accumulator.updateData(val)
    for x in eval_accumulator:
        val[x] = eval_accumulator[x]
    norm_feat_s = val['norm_feat_s']
    label_s = val['label_s']
    source_emb = norm_feat_s
    target_emb = norm_feat_t
    source_pred = label_s
    del val
    if label_t is None:
        return ({}, source_emb, source_pred, target_emb, None, {})
    cls_output_tensor = torch.from_numpy(cls_output_t).float().to(model.device)
    cls_probs = torch.softmax(cls_output_tensor, dim=1)
    (_, cls_preds) = torch.max(cls_probs, dim=1)
    cls_probs_np = cls_probs.detach().cpu().numpy()
    cls_preds_np = cls_preds.detach().cpu().numpy()
    source_prototype = model.classifier.fc.weight.data
    stopThr = 1e-06
    norm_feat_t_tensor = torch.from_numpy(norm_feat_t).to(model.device)
    (newsim, fake_size) = adaptive_filling(norm_feat_t_tensor, source_prototype, model.device, gamma, beta, fill_size=0, stopThr=stopThr, high=high)
    (_, __, ___, ____, Q_st_bar) = ubot_CCD(newsim, beta, model.device, fake_size=fake_size, fill_size=0, mode='minibatch', stopThr=stopThr, high=False, save=None)
    print(f'\n[Debug] Q_st_bar check:')
    print(f'  Shape: {Q_st_bar.shape}')
    print(f'  Row sums min/max: {Q_st_bar.sum(dim=1).min():.6f} / {Q_st_bar.sum(dim=1).max():.6f}')
    print(f'  Column sums: {Q_st_bar.sum(dim=0)}')
    print(f'  Min/Max value: {Q_st_bar.min():.6e} / {Q_st_bar.max():.6e}')
    Q_st_full = Q_st_bar.detach().cpu().numpy()
    if Q_st_full.shape[0] > norm_feat_t.shape[0]:
        real_sample_Q = Q_st_full[-norm_feat_t.shape[0]:, :]
    else:
        real_sample_Q = Q_st_full
    ot_source_probs = real_sample_Q[:, :num_source_classes]
    assigned_mass = ot_source_probs.sum(axis=1)
    N_aug = Q_st_full.shape[0]
    prior_mass = 1.0 / N_aug
    ot_dummy_mass = np.clip(1.0 - assigned_mass / prior_mass, 0.0, 1.0)
    ot_conf_scores = ot_source_probs.max(axis=1)
    print(f'✓ TOS (Eq.5) range: [{ot_dummy_mass.min():.6f}, {ot_dummy_mass.max():.6f}]')
    print(f'✓ UOT_conf (w_i) range: [{ot_conf_scores.min():.6f}, {ot_conf_scores.max():.6f}]')
    print(f'  N_aug = {N_aug}, prior_mass = {prior_mass:.6e}')
    ot_preds_common = np.argmax(ot_source_probs, axis=1)
    threshold = prior_mass
    ot_preds_with_private = ot_preds_common.copy()
    private_mask = ot_conf_scores <= threshold
    ot_preds_with_private[private_mask] = private_class_idx
    print(f'  Private samples (w_i <= {threshold:.6e}): {private_mask.sum()} / {len(private_mask)} ({private_mask.mean() * 100:.1f}%)')
    if plot:
        classes = label_mapping['cell_type'].tolist() if hasattr(label_mapping['cell_type'], 'tolist') else list(label_mapping['cell_type'])
        weights_df = pd.read_csv(config.path_output + 'wt_i.csv') if os.path.exists(config.path_output + 'wt_i.csv') else None
        target_weights_plot = list(weights_df['wt_i']) if weights_df is not None else None
        plot_separate_figures(source_feat=norm_feat_s, target_feat=norm_feat_t, source_labels=label_s, target_labels=label_t, pred_target=ot_preds_with_private, class_names=classes, save_dir=config.path_output, prefix=epoch if epoch is not None else 'final', n_samples=None, source_weights=None, target_weights=target_weights_plot)
        save_predictions(id=ids_t, true_labels=label_t.astype(int), pred_labels=ot_preds_with_private, save_dir=config.path_output)
    common_mask = label_t < num_source_classes
    acc_common = calculate_accuracy_simple(label_t, ot_preds_common, common_mask)
    label_t_unified = label_t.copy()
    label_t_unified[label_t_unified >= num_source_classes] = private_class_idx
    acc_overall = accuracy_score(label_t_unified, ot_preds_with_private)
    results_dict = {'cls_common_acc': acc_common, 'cls_overall_acc': acc_overall}
    print(f'\n[Epoch {epoch}] Evaluation:')
    print(f'- Common Acc (OT, High=False): {acc_common * 100:.2f}%')
    print(f'- Overall Acc (OT, High=True): {acc_overall * 100:.2f}%')
    save_pack = {'ids': ids_t, 'embedding': norm_feat_t, 'cls_probs': cls_probs_np, 'cls_preds': cls_preds_np, 'ot_matrix': real_sample_Q, 'ot_preds_common': ot_preds_common, 'ot_preds_private': ot_preds_with_private, 'ot_dummy_mass': ot_dummy_mass, 'ot_conf_scores': ot_conf_scores}
    model.train()
    return (results_dict, source_emb, source_pred, target_emb, ot_preds_with_private, save_pack)
