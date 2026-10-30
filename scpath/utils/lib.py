import torch
import numpy as np
import torch.nn.functional as F
import ot
from torch_geometric.data import Batch
import pandas as pd
import torch
from torch.utils.data import Dataset
from skimage.filters import threshold_otsu

def collate_fn(batch):
    x_list = []
    global_ids = []
    for item in batch:
        x = item[0]['x']
        if x.ndim == 1:
            x = x.unsqueeze(0)
        x_list.append(x)
        global_ids.append(item[1].item())
    x_batch = torch.cat(x_list, dim=0)
    ids_batch = torch.stack([item[1] for item in batch])
    global_to_local = {gid: i for (i, gid) in enumerate(global_ids)}
    pathway_data = {}
    for p in batch[0][0]['edge_indices'].keys():
        all_edges = []
        batch_vector = []
        for (batch_idx, (sample, node_id)) in enumerate(batch):
            edges = sample['edge_indices'][p]
            src_global = node_id.item()
            dst_globals = edges[1].tolist()
            valid_dst = []
            for dst_g in dst_globals:
                if dst_g in global_to_local:
                    valid_dst.append(global_to_local[dst_g])
            if not valid_dst:
                continue
            adjusted_edges = torch.stack([torch.full((len(valid_dst),), batch_idx, dtype=torch.long), torch.tensor(valid_dst, dtype=torch.long)])
            all_edges.append(adjusted_edges)
            batch_vector.append(torch.full((len(valid_dst),), batch_idx, dtype=torch.long))
        edge_index = torch.cat(all_edges, dim=1) if all_edges else torch.empty((2, 0), dtype=torch.long)
        if edge_index.numel() > 0:
            max_index = edge_index.max().item()
            assert max_index < x_batch.size(0), f'Edge index out of range: {max_index} >= {x_batch.size(0)}'
        pathway_data[p] = {'x': x_batch, 'edge_index': edge_index, 'batch': torch.cat(batch_vector) if batch_vector else torch.tensor([], dtype=torch.long)}
    return (pathway_data, ids_batch)

def seed_everything(seed=1234):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)

def sinkhorn(out, epsilon, sinkhorn_iterations, global_step=None, prior=None, eps=1e-08):
    """
    Improved Sinkhorn-Knopp algorithm with:
    - Dynamic iteration adjustment
    - Numerical stability fixes
    - Prior distribution support
    """
    out = torch.clamp(out, min=-50, max=50)
    Q = torch.exp(out / epsilon)
    (B, K) = Q.shape
    if global_step is not None:
        if global_step < 300:
            actual_iterations = min(5, sinkhorn_iterations)
        else:
            actual_iterations = sinkhorn_iterations + 5
    else:
        actual_iterations = sinkhorn_iterations
    if prior is None:
        p = torch.ones(K, device=Q.device) / K
    else:
        p = prior.clone().detach()
        p = p / p.sum()
    for it in range(actual_iterations):
        Q = Q / (Q.sum(dim=1, keepdim=True) + eps)
        Q = Q * p.unsqueeze(0) / (Q.sum(dim=0, keepdim=True) + eps)
    return Q

def ubot_CCD(sim, beta, device, fake_size=0, fill_size=0, mode='minibatch', stopThr=0.0001, high=True, save=None):
    if torch.isnan(sim).any() or torch.isinf(sim).any():
        print(f'sim contains NaN/Inf: min={sim.min()}, max={sim.max()}')
        sim = torch.nan_to_num(sim)
    M = -sim
    if beta is not None:
        class_weights = 1 / (beta + 1e-06)
        class_weights /= class_weights.sum()
    beta = torch.as_tensor(beta).to(device)
    beta_sum = beta.sum()
    if beta_sum <= 0 or beta.min() < 0:
        print(f'Invalid beta: sum={beta_sum}, min={beta.min()} - Correcting')
        beta = torch.clamp(beta, min=0)
        beta = beta / beta.sum()
    alpha = ot.unif(sim.size(0))
    M_np = M.detach().cpu().numpy()
    try:
        Q_st = ot.unbalanced.sinkhorn_knopp_unbalanced(alpha, beta.cpu().numpy(), M_np, reg=0.03, reg_m=0.5, weights=class_weights if beta is not None else None, stopThr=stopThr)
    except Exception as e:
        print(f'Sinkhorn failed: {e} - Using fallback')
        Q_st = np.ones((sim.size(0), beta.size(0))) / (sim.size(0) * beta.size(0))
    Q_st = torch.from_numpy(Q_st).float().to(device)
    sum_pi = torch.sum(Q_st)
    if sum_pi <= 0 or torch.isnan(sum_pi):
        print(f'Invalid Q_st sum: {sum_pi} - Using uniform')
        Q_st_bar = torch.ones_like(Q_st) / Q_st.numel()
    else:
        Q_st_bar = Q_st / sum_pi
    if mode == 'minibatch':
        Q_anchor = Q_st_bar[fake_size + fill_size:, :]
    if mode == 'all':
        Q_anchor = Q_st_bar
    (wt_i, pseudo_label) = torch.max(Q_anchor, 1)
    wt_i_np = wt_i.detach().cpu().numpy()
    ws_j = torch.sum(Q_st_bar, 0)
    try:
        otsu_threshold = threshold_otsu(wt_i_np)
    except ValueError:
        otsu_threshold = np.mean(wt_i_np)
    otsu_threshold = torch.tensor(otsu_threshold).float().to(device)
    if save != None:

        def save_to_csv(Q_st, filename):
            import csv
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([f'# Matrix shape: {Q_st.shape}'])
                writer.writerow([f'# Min value: {Q_st.min():.6f}, Max value: {Q_st.max():.6f}'])
                if torch.is_tensor(Q_st):
                    Q_st_np = Q_st.detach().cpu().numpy()
                else:
                    Q_st_np = Q_st
                np.savetxt(f, Q_st_np, delimiter=',', fmt='%.6f')
        df = pd.DataFrame({'wt_i': wt_i_np, 'pseudo_label': pseudo_label.detach().cpu().numpy(), 'otsu_threshold': [otsu_threshold.item()] * len(wt_i_np), '1/w': [1 / Q_st_bar.size(0)] * len(wt_i_np)})
        df.to_csv(save + 'wt_i.csv', index=False)
        Q_st_np = Q_anchor.detach().cpu().numpy()
        df = pd.DataFrame(Q_st_np)
        df.to_csv(save + 'Q_st_matrix.csv', index=False)
        pd.DataFrame(Q_st_np).to_csv(save + 'ot_matrix.csv', index=False, header=False)
        df = pd.DataFrame({'ws_j': ws_j.detach().cpu().numpy()})
        df.to_csv(save + 'ws_j.csv', index=False)
        with open(save + 'Q_st_bar_size.txt', 'w') as f:
            f.write(f'Q_st_bar:{Q_st_bar.size()}\n')
    uniformed_index = Q_st_bar.size(1)
    if high:
        conf_label = pseudo_label
        high_conf_label = torch.where(wt_i > 1 / Q_st_bar.size(0), pseudo_label, uniformed_index)
        high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)
    else:
        conf_label = pseudo_label
        high_conf_label = conf_label.clone()
        high_conf_label_id = torch.nonzero(high_conf_label != uniformed_index).view(-1)
    new_beta = torch.sum(Q_st_bar, 0).cpu().numpy()
    if np.isnan(new_beta).any():
        print('new_beta still has NaN! Using beta backup')
        new_beta = beta.cpu().numpy()
    return (high_conf_label_id, high_conf_label, conf_label, new_beta, Q_st_bar)

def adaptive_filling(ubot_feature_t, source_prototype, device, gamma, beta, fill_size, stopThr=0.0001, high=True):
    sim = torch.matmul(ubot_feature_t, source_prototype.t())
    (max_sim, _) = torch.max(sim, 1)
    pos_id = torch.nonzero(max_sim > gamma).reshape(-1)
    pos_rate = pos_id.size(0) / max_sim.size(0)
    pos_num = pos_id.size(0)
    neg_num = max_sim.size(0) - pos_num
    if pos_rate <= 0.5:
        fake_size = neg_num - pos_num
        if fake_size > 0:
            (high_conf_label_id, high_conf_label, __, ___, ____) = ubot_CCD(sim, beta, device, fake_size=0, fill_size=fill_size, mode='all', stopThr=stopThr, high=high)
            if high_conf_label_id.size(0) > 0:
                class_weights = torch.ones_like(torch.tensor(beta)).to(device)
                for cls_idx in range(beta.shape[0]):
                    cls_mask = high_conf_label[high_conf_label_id] == cls_idx
                    if cls_mask.sum() > 0:
                        freq = cls_mask.float().mean()
                        class_weights[cls_idx] = 1 / (freq.item() + 1e-06)
                class_weights = class_weights / class_weights.sum()
                sample_weights = class_weights[high_conf_label[high_conf_label_id]]
                sample_weights = sample_weights / sample_weights.sum()
                select_id = torch.multinomial(sample_weights, fake_size, replacement=True)
                fill_pos = sim[high_conf_label_id[select_id], :]
                newsim = torch.cat([fill_pos, sim], 0)
            else:
                fake_size = 0
                newsim = sim
        else:
            newsim = sim
    else:
        fake_size = pos_num - neg_num
        if fake_size > 0:
            farthest_sproto_id = torch.argmin(sim, 1)
            fake_private = 0.5 * ubot_feature_t + 0.5 * source_prototype.data[farthest_sproto_id, :]
            fake_private = F.normalize(fake_private)
            select_id = torch.randint(0, fake_private.size(0), (fake_size,)).to(device)
            fill_neg = fake_private[select_id, :]
            fake_sim = torch.matmul(fill_neg, source_prototype.t())
            newsim = torch.cat([fake_sim, sim], 0)
        else:
            newsim = sim
    return (newsim, fake_size)
