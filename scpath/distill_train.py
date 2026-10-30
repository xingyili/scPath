import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pandas as pd
import os
import scanpy as sc
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from scpath.utils.lib import seed_everything

class DistillConfig:

    def __init__(self, original_config):
        self.device = original_config.device
        self.log_dir = original_config.log_dir
        self.batch_size = max(min(256, original_config.batch_size), original_config.batch_size // 2)
        self.epochs = 200
        self.lr = 0.001
        self.seed = 42
        self.anchor_percentile = 0.6
        self.min_samples_threshold = 5
        self.private_thr = 0.7

class SimpleStudent(nn.Module):

    def __init__(self, input_dim, num_classes, hidden_dim=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim // 2), nn.BatchNorm1d(hidden_dim // 2), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, num_classes))

    def forward(self, x):
        return self.net(x)

def prepare_distill_data(config, adata_test_path):
    print(f'Loading data from: {adata_test_path}')
    adata = sc.read_h5ad(adata_test_path)
    features = adata.obsm['X_emb']
    ot_matrix = adata.obsm['ot_matrix']
    dummy_mass = adata.obs['ot_dummy_mass'].values
    conf_scores = adata.obs['ot_conf_scores'].values
    true_labels = adata.obs['labels'].values
    teacher_preds = adata.obs['pred_labels_ot_private'].values
    ot_preds_common = adata.obs['pred_labels_ot_common'].values
    num_classes = adata.obsm['cls_probs'].shape[1]
    unified_private_label_id = num_classes
    ot_source_part = ot_matrix[:, :num_classes]
    soft_labels = ot_source_part / (ot_source_part.sum(axis=1, keepdims=True) + 1e-08)
    mask_anchor = np.zeros(len(features), dtype=bool)
    mask_private = np.zeros(len(features), dtype=bool)
    candidate_preds = ot_preds_common
    print('\n[Class-Balanced Anchor Selection]')
    for k in range(num_classes):
        k_indices = np.where(candidate_preds == k)[0]
        if len(k_indices) == 0:
            print(f'  - Class {k}: No samples predicted.')
            continue
        k_scores = conf_scores[k_indices]
        n_k = len(k_indices)
        if n_k <= config.min_samples_threshold:
            is_anchor = candidate_preds == k
            print(f'  - Class {k}: {n_k} candidates (<= {config.min_samples_threshold}) -> All Selected as Anchors (Rare Class Protection)')
        else:
            k_thr = np.quantile(k_scores, 1 - config.anchor_percentile)
            is_anchor = (candidate_preds == k) & (conf_scores >= k_thr) & (dummy_mass < config.private_thr)
            print(f'  - Class {k}: {n_k} candidates -> {np.sum(is_anchor)} Anchors (Thr={k_thr:.6f})')
        mask_anchor = mask_anchor | is_anchor
    mask_private = dummy_mass > config.private_thr
    mask_boundary = ~(mask_anchor | mask_private)
    print(f'\n[Global Stats] N={len(features)}')
    print(f' - Anchor Samples:   {sum(mask_anchor)} (balanced across common classes)')
    print(f' - Boundary Samples: {sum(mask_boundary)} (require optimization)')
    print(f' - Private Samples:  {sum(mask_private)}')
    data_tensors = {'features': torch.FloatTensor(features), 'soft_labels': torch.FloatTensor(soft_labels), 'mask_anchor': torch.BoolTensor(mask_anchor), 'mask_boundary': torch.BoolTensor(mask_boundary), 'mask_private': torch.BoolTensor(mask_private)}
    meta_info = {'num_classes': num_classes, 'unified_private_label_id': unified_private_label_id, 'true_labels': true_labels, 'teacher_preds': teacher_preds, 'ot_preds_common': ot_preds_common, 'dummy_mass': dummy_mass}
    return (data_tensors, meta_info)

def train_student_model(config, data_tensors, num_classes):
    seed_everything(config.seed)
    dataset = TensorDataset(data_tensors['features'], data_tensors['soft_labels'], data_tensors['mask_anchor'], data_tensors['mask_boundary'])
    print(f'distill batch size:{config.batch_size}')
    last_batch_remainder = len(dataset) % config.batch_size
    drop_last_flag = last_batch_remainder == 1
    print(f'Data N={len(dataset)}, Batch={config.batch_size}, Remainder={last_batch_remainder} => Drop Last: {drop_last_flag}')
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, drop_last=drop_last_flag)
    student = SimpleStudent(input_dim=data_tensors['features'].shape[1], num_classes=num_classes).to(config.device)
    optimizer = optim.Adam(student.parameters(), lr=config.lr, weight_decay=0.0001)
    student.train()
    print(f'\nStarting training for {config.epochs} epochs...')
    for epoch in range(config.epochs):
        total_loss = 0
        for (feat, soft_y, m_anc, m_bnd) in dataloader:
            (feat, soft_y, m_anc, m_bnd) = [x.to(config.device) for x in [feat, soft_y, m_anc, m_bnd]]
            logits = student(feat)
            loss = 0
            if m_anc.any():
                loss += nn.KLDivLoss(reduction='batchmean')(F.log_softmax(logits[m_anc], dim=1), soft_y[m_anc]) * 1.0
            if m_bnd.any():
                probs = F.softmax(logits[m_bnd], dim=1)
                log_probs = F.log_softmax(logits[m_bnd], dim=1)
                entropy = -(probs * log_probs).sum(dim=1).mean()
                loss += entropy * 0.5
            if isinstance(loss, torch.Tensor):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
    print('Training finished.')
    return student

def run_reclamation(config, student, data_tensors, meta_info):
    student.eval()
    with torch.no_grad():
        features = data_tensors['features'].to(config.device)
        logits = student(features)
        probs = F.softmax(logits, dim=1)
        (student_conf, student_preds) = torch.max(probs, dim=1)
        student_preds = student_preds.cpu().numpy()
        student_conf = student_conf.cpu().numpy()
    teacher_preds = meta_info['teacher_preds']
    dummy_mass = meta_info['dummy_mass']
    num_classes = meta_info['num_classes']
    unified_private_label_id = meta_info['unified_private_label_id']
    final_preds = teacher_preds.copy()
    reclaimed_mask = np.zeros(len(final_preds), dtype=bool)
    for i in range(len(final_preds)):
        if final_preds[i] < num_classes:
            continue
        elif final_preds[i] == unified_private_label_id:
            if dummy_mass[i] > config.private_thr:
                continue
            else:
                final_preds[i] = student_preds[i]
                reclaimed_mask[i] = True
    return (final_preds, student_preds, reclaimed_mask)

def evaluate_and_plot(meta_info, final_preds, student_preds, save_dir):
    true_labels = np.array(meta_info['true_labels']).astype(int)
    teacher_preds = meta_info['teacher_preds']
    ot_preds_common = meta_info['ot_preds_common']
    num_classes = meta_info['num_classes']
    unified_private_label_id = meta_info['unified_private_label_id']

    def unify_labels_for_acc(lbls):
        l = lbls.copy()
        l[l >= num_classes] = unified_private_label_id
        return l
    true_unified_for_acc = unify_labels_for_acc(true_labels)
    acc_teacher = accuracy_score(true_unified_for_acc, teacher_preds)
    acc_high_false = accuracy_score(true_unified_for_acc, ot_preds_common)
    acc_student = accuracy_score(true_unified_for_acc, student_preds)
    acc_final = accuracy_score(true_unified_for_acc, final_preds)
    print('\n================ Final results comparison ================')
    print(f'1. Teacher (High=True) Acc:  {acc_teacher * 100:.2f}%  (includes private-class recognition)')
    print(f'2. OT Baseline (High=False): {acc_high_false * 100:.2f}% (no private-class recognition)')
    print(f'3. Pure Student Acc:         {acc_student * 100:.2f}% (no private-class recognition)')
    print(f'--------------------------------------------')
    print(f'4. Final Reclaimed Acc:      {acc_final * 100:.2f}%  <-- target')
    print('============================================')
    if acc_final >= acc_high_false:
        print('Target achieved: accuracy meets or exceeds High=False while retaining private-class recognition.')
    else:
        print('Warning: accuracy is below High=False; the threshold may be too strict or student training may be insufficient.')
    n_private_teacher = np.sum(teacher_preds == unified_private_label_id)
    n_private_final = np.sum(final_preds == unified_private_label_id)
    print(f'\nPrivate-class statistics:')
    print(f' - Teacher decisions: {n_private_teacher}')
    print(f' - Final decisions:   {n_private_final}')
    print(f' - Reclaimed:     {n_private_teacher - n_private_final} (corrected by the student)')
    res_df = pd.DataFrame({'true': true_labels, 'teacher_high_true': teacher_preds, 'ot_high_false': ot_preds_common, 'final_pred': final_preds, 'is_reclaimed': teacher_preds != final_preds})
    res_df.to_csv(os.path.join(save_dir, 'final_distill_analysis.csv'), index=False)
    return acc_final

def update_and_save_adata(config, test_file, final_preds, student_preds, reclaimed_mask):
    print(f'\nUpdating AnnData from: {test_file}')
    adata = sc.read_h5ad(test_file)
    adata.obs['final_distill_preds'] = final_preds
    adata.obs['student_preds'] = student_preds
    adata.obs['is_reclaimed'] = reclaimed_mask
    save_path = os.path.join(config.log_dir, 'test_results_distilled.h5ad')
    print(f'Saving updated AnnData to: {save_path}')
    adata.write(save_path)
    return adata
