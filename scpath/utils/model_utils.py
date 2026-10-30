import torch
from collections import OrderedDict
from scpath.models import DomainAdaptFusionModel
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset
import scipy.sparse

class GraphIDDataset(Dataset):

    def __init__(self, adata, graphs, data_type, pathway_names):
        self.adata = adata
        self.graphs = graphs
        self.data_type = data_type
        self.pathway_names = pathway_names
        self.int_ids = torch.LongTensor(adata.obs['int_id'].values)
        if scipy.sparse.issparse(adata.X):
            self.all_features = torch.FloatTensor(adata.X.toarray())
        else:
            self.all_features = torch.FloatTensor(adata.X)

    def __len__(self):
        return len(self.adata)

    def __getitem__(self, idx):
        node_id = self.int_ids[idx]
        x = self.all_features[idx]
        if x.ndim == 1:
            x = x.unsqueeze(0)
        edge_indices = {}
        for p in self.pathway_names:
            neighbors = self.graphs[f'{self.data_type}_{p}'][idx].indices
            edge_index = torch.stack([torch.full((len(neighbors),), node_id, dtype=torch.long), torch.tensor(neighbors, dtype=torch.long)])
            edge_indices[p] = edge_index
        return ({'x': x, 'edge_indices': edge_indices}, node_id)

class MemoryQueue(nn.Module):

    def __init__(self, feat_dim, batchsize, n_batch, device, T=0.05):
        super(MemoryQueue, self).__init__()
        self.feat_dim = feat_dim
        self.batchsize = batchsize
        self.T = T
        self.device = device
        self.queue_size = self.batchsize * n_batch
        self.register_buffer('mem_feat', torch.zeros(self.queue_size, feat_dim))
        self.register_buffer('mem_id', torch.zeros(self.queue_size, dtype=int))
        self.mem_feat = self.mem_feat.to(self.device)
        self.mem_id = self.mem_id.to(self.device)
        self.next_write = 0

    def __repr__(self):
        return f'MemoryQueue(queue_size={self.queue_size}, next_write={self.next_write}, mem_feat_shape={self.mem_feat}, mem_id_shape={self.mem_id})'

    def forward(self, x):
        """
        obtain similarity between x and the features stored in memory queue
        """
        out = torch.mm(x, self.mem_feat.t()) / self.T
        return out

    def get_nearest_neighbor(self, anchors, id_anchors=None):
        """
        get anchors' nearest neighbor in memory queue 
        """
        feat_mat = self.forward(anchors)
        if id_anchors is not None:
            A = id_anchors.reshape(-1, 1).repeat(1, self.mem_id.size(0))
            B = self.mem_id.reshape(1, -1).repeat(id_anchors.size(0), 1)
            mask = torch.eq(A, B)
            id_mask = torch.nonzero(mask)
            temp = id_mask[:, 1]
            feat_mat[:, temp] = -1 / self.T
        (values, indices) = torch.max(feat_mat, 1)
        nearest_feat = torch.zeros((anchors.size(0), self.feat_dim)).to(self.device)
        for i in range(anchors.size(0)):
            nearest_feat[i] = self.mem_feat[indices[i], :]
        return (values, nearest_feat)

    def update_queue(self, features, ids):
        """
        update memory queue
        """
        w_ids = torch.arange(self.next_write, self.next_write + self.batchsize).to(self.device)
        self.mem_feat.index_copy_(0, w_ids, features.data)
        self.mem_id.index_copy_(0, w_ids, ids.data)
        self.mem_feat = F.normalize(self.mem_feat)
        self.next_write += self.batchsize
        if self.next_write == self.queue_size:
            self.next_write = 0

    def random_sample(self, size):
        """
        sample some features from memory queue randomly
        """
        id_t = torch.floor(torch.rand(size) * self.mem_feat.size(0)).long().to(self.device)
        sample_feat = self.mem_feat[id_t]
        return sample_feat

def load_fusion_model(checkpoint_path, base_models, config, device='cuda'):
    fusion_model = DomainAdaptFusionModel(base_models, config.num_classes, config).to(device)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if 'base_models' in checkpoint:
            for (name, state_dict) in checkpoint['base_models'].items():
                if name in base_models:
                    base_models[name].load_state_dict(state_dict)
                else:
                    print(f'Warning: checkpoint base model {name} is absent from the current configuration')
        fusion_model.bottleneck.load_state_dict(checkpoint.get('bottleneck', OrderedDict()))
        fusion_model.classifier.load_state_dict(checkpoint.get('classifier', OrderedDict()))
        fusion_model.discriminator.load_state_dict(checkpoint.get('discriminator', OrderedDict()))
        fusion_model.discriminator_separate.load_state_dict(checkpoint.get('discriminator_separate', OrderedDict()))
        if 'config' in checkpoint:
            for (k, v) in checkpoint['config'].items():
                if hasattr(config, k):
                    setattr(config, k, v)
        print('Model loaded successfully!')
        return fusion_model
    except Exception as e:
        print(f'Loading failed: {str(e)}')
        return fusion_model
