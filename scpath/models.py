import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, SAGEConv
import torch.nn as nn
import numpy as np
from torch.autograd import Function

class GradientReverseLayer(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return (output, None)

def grad_reverse(x, alpha=1.0):
    return GradientReverseLayer.apply(x, alpha)

class DomainDiscriminator(nn.Module):

    def __init__(self, in_dim, hidden_dim=256):
        super(DomainDiscriminator, self).__init__()
        self.layer = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, x, alpha):
        x = grad_reverse(x, alpha)
        return self.layer(x)

class MPNNs(torch.nn.Module):
    """Encode a cell graph and fuse layer features with the source MLP gate."""

    def __init__(self, in_channels, hidden_channels, out_channels, local_layers=3, dropout=0.5, heads=1, pre_ln=False, pre_linear=False, res=False, ln=False, bn=False, jk=False, gnn='gcn'):
        super(MPNNs, self).__init__()
        self.dropout = dropout
        self.pre_ln = pre_ln
        self.total_layers = local_layers
        self.pre_linear = pre_linear
        self.res = res
        self.ln = ln
        self.bn = bn
        self.jk = jk
        self.hidden_channels = hidden_channels
        self.h_lins = torch.nn.ModuleList()
        self.local_convs = torch.nn.ModuleList()
        self.lins = torch.nn.ModuleList()
        self.lns = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if self.pre_ln:
            self.pre_lns = torch.nn.ModuleList()
        self.lin_in = torch.nn.Linear(in_channels, hidden_channels)
        if not self.pre_linear:
            if gnn == 'gat':
                self.local_convs.append(GATConv(in_channels, hidden_channels // heads, heads=heads, concat=True, add_self_loops=False, bias=False))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(in_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(in_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(in_channels))
            local_layers = local_layers - 1
        for _ in range(local_layers):
            if gnn == 'gat':
                self.local_convs.append(GATConv(hidden_channels, hidden_channels // heads, heads=heads, concat=True, add_self_loops=False, bias=False))
            elif gnn == 'sage':
                self.local_convs.append(SAGEConv(hidden_channels, hidden_channels))
            else:
                self.local_convs.append(GCNConv(hidden_channels, hidden_channels, cached=False, normalize=True))
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            self.lns.append(torch.nn.LayerNorm(hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            if self.pre_ln:
                self.pre_lns.append(torch.nn.LayerNorm(hidden_channels))
        self.pred_local = torch.nn.Linear(hidden_channels, out_channels)
        gate_dim = hidden_channels * self.total_layers
        self.scale_gate = nn.Sequential(nn.Linear(gate_dim, gate_dim), nn.ReLU(), nn.Linear(gate_dim, gate_dim), nn.Sigmoid())
        self.residual_proj = nn.ModuleList()
        num_layers = len(self.local_convs)
        for _ in range(num_layers):
            self.residual_proj.append(nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity())
        self.residual_final_proj = nn.Linear(in_channels, hidden_channels) if in_channels != hidden_channels else nn.Identity()

    def reset_parameters(self):
        for local_conv in self.local_convs:
            local_conv.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for ln in self.lns:
            ln.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        if self.pre_ln:
            for p_ln in self.pre_lns:
                p_ln.reset_parameters()
        self.lin_in.reset_parameters()
        self.pred_local.reset_parameters()
        for layer in self.scale_gate:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for proj_layer in self.residual_proj:
            if hasattr(proj_layer, 'reset_parameters'):
                proj_layer.reset_parameters()
        if hasattr(self.residual_final_proj, 'reset_parameters'):
            self.residual_final_proj.reset_parameters()

    def forward(self, x, edge_index):
        inputs_initial = x.clone()
        if self.pre_linear:
            x = self.lin_in(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x_final = 0
        multi_scale_features = []
        for (i, local_conv) in enumerate(self.local_convs):
            if self.res:
                residual = self.residual_proj[i - 1](inputs_initial) if hasattr(self, 'residual_proj') else inputs_initial
                x = local_conv(x, edge_index) + self.lins[i](x) + residual
            else:
                x = local_conv(x, edge_index)
            if self.ln:
                x = self.lns[i](x)
            elif self.bn:
                x = self.bns[i](x)
            else:
                pass
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if self.jk:
                x_final = x_final + x
            else:
                x_final = x
            multi_scale_features.append(x)
        if len(multi_scale_features) > 1:
            concat_features = torch.cat(multi_scale_features, dim=1)
            expected_dim = self.hidden_channels * self.total_layers
            if concat_features.size(1) != expected_dim:
                print(f'Warning: concatenated dimension {concat_features.size(1)} != expected dimension {expected_dim}')
            gate_weights = torch.sigmoid(self.scale_gate(concat_features))
            split_sizes = [feat.size(1) for feat in multi_scale_features]
            gate_splits = torch.split(gate_weights, split_sizes, dim=1)
            fused = sum((gate * feat for (gate, feat) in zip(gate_splits, multi_scale_features)))
        else:
            fused = multi_scale_features[0] if multi_scale_features else x
        x_final = fused
        x_final = F.relu(x_final)
        return (x_final, self.pred_local(x_final))

class DynamicGate(nn.Module):

    def __init__(self, feat_dim=256, out_dim=3, num_classes=8):
        super().__init__()
        self.out_dim = out_dim
        self.num_classes = num_classes
        self.class_embed_dim = 16
        self.gru = nn.GRU(feat_dim + self.class_embed_dim, 64, batch_first=True)
        self.gate_net = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, out_dim), nn.Softmax(dim=-1))
        self.class_embed = nn.Embedding(num_classes, self.class_embed_dim)

    def forward(self, features, class_labels=None, class_weights=None):
        batch_size = features.size(0)
        if class_labels is not None:
            class_emb = self.class_embed(class_labels.long())
        elif batch_size > 0:
            uniform_labels = torch.randint(0, self.num_classes, (batch_size,), device=features.device)
            class_emb = self.class_embed(uniform_labels)
        else:
            class_emb = torch.zeros((0, self.class_embed_dim), device=features.device)
        if class_weights is not None:
            weight_vector = class_weights[class_labels] if class_labels is not None else class_weights.mean()
            class_emb = class_emb * weight_vector.view(-1, 1)
        features_with_class = torch.cat([features, class_emb], dim=1)
        if features_with_class.size(0) > 0:
            features_with_class = features_with_class.unsqueeze(1)
            _, hidden = self.gru(features_with_class)
            return self.gate_net(hidden.squeeze(0))
        else:
            return torch.zeros((0, self.out_dim), device=features.device)

class ProtoCLS(nn.Module):
    """
    prototype-based classifier
    L2-norm + a fc layer (without bias)
    """

    def __init__(self, in_dim, out_dim, temp=0.05):
        super(ProtoCLS, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        self.tmp = temp
        self.weight_norm()

    def forward(self, x):
        x = F.normalize(x)
        assert x.shape[1] == self.fc.weight.shape[1], 'Feature dimensions do not match the classifier!'
        x = self.fc(x) / self.tmp
        return x

    def weight_norm(self):
        w = self.fc.weight.data
        norm = w.norm(p=2, dim=1, keepdim=True) + 1e-08
        self.fc.weight.data = w.div(norm.expand_as(w))

class ReconstructionLoss(nn.Module):

    def __init__(self, pathways, base_models, loss_type='mse', alpha=1.0, beta=1.0, gamma=0.5):
        super().__init__()
        self.pathways = pathways
        self.base_models = base_models
        self.loss_type = loss_type
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        if loss_type == 'mse':
            self.loss_fn = nn.MSELoss()
        elif loss_type == 'mae':
            self.loss_fn = nn.L1Loss()
        elif loss_type == 'cosine':
            self.loss_fn = nn.CosineEmbeddingLoss()
        else:
            raise ValueError(f'Unsupported loss type: {loss_type}')

    def forward(self, model_output, original_inputs):
        losses = {}
        total_loss = 0
        path_losses = []
        for path in self.pathways:
            if path in model_output.get('reconstructions', {}) and path in original_inputs:
                x_orig = original_inputs[path][0]
                x_recon = model_output['reconstructions'][path]
                if self.loss_type == 'cosine':
                    labels = torch.ones(x_orig.size(0), device=x_orig.device)
                    loss = self.loss_fn(x_orig, x_recon, labels)
                else:
                    loss = self.loss_fn(x_orig, x_recon)
                path_losses.append(loss)
                losses[f'recon_path_{path}'] = loss.item()
        if path_losses:
            path_loss_avg = sum(path_losses) / len(path_losses)
            total_loss += self.alpha * path_loss_avg
            losses['recon_path_avg'] = path_loss_avg.item()
        if 'global_reconstruction' in model_output:
            x_orig_global = torch.cat([original_inputs[path][0] for path in self.pathways], dim=1)
            x_recon_global = model_output['global_reconstruction']
            if self.loss_type == 'cosine':
                labels = torch.ones(x_orig_global.size(0), device=x_orig_global.device)
                global_loss = self.loss_fn(x_orig_global, x_recon_global, labels)
            else:
                global_loss = self.loss_fn(x_orig_global, x_recon_global)
            total_loss += self.beta * global_loss
            losses['recon_global'] = global_loss.item()
        if self.gamma > 0 and 'path_features' in model_output and ('reconstructions' in model_output):
            consistency_loss = 0
            path_count = 0
            for path in self.pathways:
                if path in model_output['path_features'] and path in model_output['reconstructions'] and (path in original_inputs):
                    orig_emb = model_output['path_features'][path]
                    x_recon = model_output['reconstructions'][path]
                    edge_index = original_inputs[path][1]
                    with torch.no_grad():
                        (recon_emb, _) = self.base_models[path](x_recon, edge_index)
                    path_loss = F.mse_loss(orig_emb, recon_emb)
                    consistency_loss += path_loss
                    path_count += 1
            if path_count > 0:
                consistency_loss /= path_count
                total_loss += self.gamma * consistency_loss
                losses['feature_consistency'] = consistency_loss.item()
        losses['total_recon_loss'] = total_loss.item()
        return (losses, total_loss)

class ReconstructionModule(nn.Module):

    def __init__(self, input_dim, output_dim, pathway_names, hidden_dim=256, num_layers=2, dropout=0.2):
        super(ReconstructionModule, self).__init__()
        self.pathway_names = pathway_names
        self.decoders = nn.ModuleDict()
        self.output_dim = output_dim
        for path in pathway_names:
            self.decoders[path] = self._make_decoder(input_dim, hidden_dim, output_dim, num_layers, dropout)
        total_input_dim = len(pathway_names) * input_dim
        total_output_dim = len(pathway_names) * output_dim
        self.global_decoder = self._make_decoder(total_input_dim, hidden_dim, total_output_dim, num_layers, dropout)

    def _make_decoder(self, input_dim, hidden_dim, output_dim, num_layers, dropout):
        layers = []
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.GELU())
        layers.append(nn.Dropout(dropout))
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dim, output_dim))
        return nn.Sequential(*layers)

    def forward(self, path_features):
        reconstructions = {}
        for path in self.pathway_names:
            reconstructions[path] = self.decoders[path](path_features[path])
        all_features = torch.cat([path_features[path] for path in self.pathway_names], dim=1)
        global_recon = self.global_decoder(all_features)
        return (reconstructions, global_recon)

class DomainAdaptFusionModel(nn.Module):
    """Fuse pathway encoders and expose the original source/target outputs."""

    def __init__(self, base_models, num_classes, config, class_weights=None):
        super().__init__()
        self.config = config
        self.device = config.device
        self.num_classes = num_classes
        self.pathway_names = list(base_models.keys())
        print(len(self.pathway_names))
        self.base_models = nn.ModuleDict(base_models)
        if class_weights is not None:
            if isinstance(class_weights, np.ndarray):
                class_weights = torch.from_numpy(class_weights).float()
            self.register_buffer('class_weights', class_weights.to(config.device))
        else:
            self.register_buffer('class_weights', torch.ones(num_classes, device=config.device) / num_classes)
        assert (self.class_weights > 0).all(), 'Class weights must be positive!'
        self.feature_transform = nn.ModuleDict({p: self._make_transform_layer(config.base_hidden_dim) for p in self.pathway_names})
        self.feature_norm = nn.LayerNorm(config.base_hidden_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=config.base_hidden_dim, num_heads=4, dropout=0.1, batch_first=True)
        gate_input_dim = config.base_hidden_dim * len(self.pathway_names)
        self.gate = DynamicGate(feat_dim=gate_input_dim, out_dim=len(self.pathway_names), num_classes=num_classes)
        self.class_aware_attention = nn.Sequential(nn.Linear(config.base_hidden_dim, config.base_hidden_dim), nn.Tanh(), nn.Linear(config.base_hidden_dim, 1, bias=False))
        self.feature_selector = nn.Sequential(nn.Linear(config.base_hidden_dim, config.base_hidden_dim), nn.ReLU(), nn.Linear(config.base_hidden_dim, config.base_hidden_dim), nn.Sigmoid())
        self.bottleneck = nn.Sequential(nn.Linear(config.base_hidden_dim, config.fusion_hidden_dim), nn.ReLU(inplace=True), nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim))
        self.classifier = ProtoCLS(config.fusion_hidden_dim, num_classes, temp=config.temp)
        self.cluster_head = ProtoCLS(config.fusion_hidden_dim, config.K, temp=config.temp)
        first_pathway = next(iter(base_models.keys()))
        if base_models[first_pathway].pre_linear:
            original_feat_dim = base_models[first_pathway].lin_in.in_features
        else:
            original_feat_dim = base_models[first_pathway].local_convs[0].in_channels
        self.reconstruction_module = ReconstructionModule(input_dim=config.base_hidden_dim, output_dim=original_feat_dim, pathway_names=self.pathway_names, hidden_dim=config.recon_hidden_dim if hasattr(config, 'recon_hidden_dim') else 1024, num_layers=config.recon_layers if hasattr(config, 'recon_layers') else 2, dropout=config.recon_dropout if hasattr(config, 'recon_dropout') else 0.2)
        self.enhance_integration = getattr(config, 'enhance_integration', False)
        if self.enhance_integration:
            print('Adversarial Training Enabled')
            self.domain_discriminator = DomainDiscriminator(config.fusion_hidden_dim, hidden_dim=256)

    def _make_transform_layer(self, dim):
        return nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.LayerNorm(dim * 2), nn.Linear(dim * 2, dim), nn.Dropout(0.1))

    def forward(self, inputs, mode='source', original_inputs=None, class_labels=None):
        device_check = next(self.parameters()).device
        for (pathway, (x, edge_index)) in inputs.items():
            x = x.to(device_check)
            edge_index = edge_index.to(device_check)
        transformed_features = []
        raw_features = {}
        for (pathway, (x, edge_index)) in inputs.items():
            (emb, _) = self.base_models[pathway](x, edge_index)
            raw_features[pathway] = emb
            transformed = self.feature_norm(self.feature_transform[pathway](emb))
            transformed_features.append(transformed)
        recon_output = None
        if original_inputs is not None:
            recon_output = self.reconstruction_module(raw_features)
        features_stack = torch.stack(transformed_features, dim=1)
        (attn_output, _) = self.cross_attn(features_stack, features_stack, features_stack)
        attn_feat = attn_output.mean(dim=1)
        gate_input = torch.cat(transformed_features, dim=1)
        if mode == 'source':
            gate_scores = self.gate(gate_input, class_labels=class_labels, class_weights=self.class_weights)
        else:
            gate_scores = self.gate(gate_input, class_weights=self.class_weights)
        gated_features = torch.sum(features_stack * gate_scores.unsqueeze(-1), dim=1)
        class_attn = self.class_aware_attention(gated_features).squeeze(-1)
        safe_weights = self.class_weights + 1e-06
        log_weights = torch.log(safe_weights.clamp(min=1e-06, max=1000000.0))
        global_weights = torch.sigmoid(log_weights).mean()
        weighted_feature = gated_features * (class_attn.unsqueeze(-1) + global_weights)
        fused_feature = weighted_feature + attn_feat
        selected_features = fused_feature
        bottleneck_out = self.bottleneck(selected_features)
        norm_feat = F.normalize(bottleneck_out, p=2, dim=1)
        cls_out = self.classifier(bottleneck_out)
        if mode == 'target':
            cluster_out = self.cluster_head(bottleneck_out)
            output_dict = {'cls_output': cls_out, 'after_cluhead': cluster_out, 'feature': bottleneck_out, 'norm_feat': norm_feat, 'path_features': raw_features}
        elif mode == 'source':
            output_dict = {'cls_output': cls_out, 'feature': bottleneck_out, 'norm_feat': norm_feat, 'path_features': raw_features}
        if recon_output is not None:
            output_dict['reconstructions'] = recon_output[0]
            output_dict['global_reconstruction'] = recon_output[1]
            output_dict['original_inputs'] = original_inputs
        return output_dict
