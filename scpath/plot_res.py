from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import pandas as pd
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from sklearn.metrics import confusion_matrix
from matplotlib import rcParams
import textwrap
from sklearn.decomposition import PCA

def set_nature_style():
    """Configure plotting style for Nature publications with 30 distinct colors"""
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Helvetica']
    rcParams['font.size'] = 10
    rcParams['axes.linewidth'] = 0.8
    rcParams['xtick.major.size'] = 4
    rcParams['ytick.major.size'] = 4
    rcParams['xtick.minor.size'] = 2
    rcParams['ytick.minor.size'] = 2
    rcParams['legend.frameon'] = False
    rcParams['legend.fontsize'] = 9
    rcParams['axes.spines.top'] = False
    rcParams['axes.spines.right'] = False
    return ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2', '#59A14F', '#EDC948', '#B07AA1', '#FF9DA7', '#9C755F', '#BAB0AC', '#8CD17D', '#499894', '#86BCB6', '#F1CE63', '#D37295', '#A0CBE8', '#FABFD2', '#B6992D', '#D4A6C8', '#79706E', '#D7B5A6', '#9D7660', '#D7B5A6', '#D37295', '#86BCB6', '#00A6ED', '#7FB800', '#FFB400', '#F6511D', '#16488B']

def plot_separate_figures(source_feat, target_feat, source_labels, target_labels, pred_target, class_names, save_dir, prefix, n_samples=None, source_weights=None, target_weights=None):
    """
    Generate all plots separately - supports full data visualization
    
    Parameters:
        n_samples: 
            - None: use all data
            - int: sample this number of points
        source_weights: array of weights for source domain samples
        target_weights: array of weights for target domain samples
    """
    nature_colors = set_nature_style()
    os.makedirs(save_dir, exist_ok=True)
    combined_feat = np.concatenate([source_feat, target_feat], axis=0)
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    combined_tsne = tsne.fit_transform(combined_feat)
    source_tsne = combined_tsne[:len(source_feat)]
    target_tsne = combined_tsne[len(source_feat):]
    true_class_mapping = {i: name for (i, name) in enumerate(class_names)}
    private_classes = np.setdiff1d(np.unique(pred_target), np.unique(source_labels))
    pred_class_mapping = true_class_mapping.copy()
    for cls in private_classes:
        pred_class_mapping[cls] = f'Private_{cls}'
    if -1 in np.unique(target_labels):
        true_class_mapping[-1] = 'Unknown'
        pred_class_mapping[-1] = 'Unknown'
    plot_fusion_distribution(source_tsne, target_tsne, source_labels, target_labels, save_dir, prefix, nature_colors, n_samples)
    plot_true_distribution(source_tsne, source_labels, class_names, true_class_mapping, save_dir, 'source', nature_colors, n_samples)
    plot_true_distribution(target_tsne, target_labels, class_names, true_class_mapping, save_dir, 'target', nature_colors, n_samples)
    plot_target_pred_distribution(target_tsne, pred_target, class_names, pred_class_mapping, private_classes, save_dir, prefix, nature_colors, n_samples)
    plot_confusion_matrix(target_labels, pred_target, class_names, true_class_mapping, save_dir, prefix)
    plot_sankey_diagram(target_labels, pred_target, class_names, private_classes, save_dir, prefix, nature_colors)
    if source_weights is not None:
        plot_weight_scatter(source_tsne, source_weights, 'Source Domain Weight Distribution', save_dir, f'{prefix}_source_weights_scatter' if prefix else 'source_weights_scatter', cmap='viridis', log_scale=True)
    if target_weights is not None:
        plot_weight_scatter(target_tsne, target_weights, 'Target Domain Weight Distribution', save_dir, f'{prefix}_target_weights_scatter' if prefix else 'target_weights_scatter', cmap='viridis', log_scale=True)

def plot_weight_scatter(coords, weights, title, save_dir, filename, cmap='viridis', log_scale=False):
    plt.figure(figsize=(10, 8), dpi=300)
    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=weights, cmap=cmap, s=15, alpha=0.7, edgecolors='none')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Weight Value', fontsize=10)
    if log_scale:
        scatter.set_norm(mcolors.LogNorm())
        cbar.set_label('Weight Value (log scale)', fontsize=10)
    plt.title(title, fontsize=12)
    plt.xlabel('Dimension 1', fontsize=10)
    plt.ylabel('Dimension 2', fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{filename}.png'), dpi=300)
    plt.close()

def sample_if_needed(data, n_samples):
    """Sample data if needed, otherwise return full data"""
    if n_samples is not None and len(data) > n_samples:
        indices = np.random.choice(len(data), n_samples, replace=False)
        return data[indices]
    return data

def plot_feature_distribution(source_coords, target_coords, source_labels, target_labels, save_dir, prefix, n_samples=None):
    colors = set_nature_style()
    plt.figure(figsize=(8, 6), dpi=300)
    source_coords = sample_if_needed(source_coords, n_samples)
    target_coords = sample_if_needed(target_coords, n_samples)
    plt.scatter(source_coords[:, 0], source_coords[:, 1], color=colors[0], alpha=0.6, s=15, label='Training Set')
    plt.scatter(target_coords[:, 0], target_coords[:, 1], color=colors[1], marker='x', alpha=0.7, s=15, label='Test Set')
    plt.title(f'Original Feature Distribution: {prefix}', fontsize=12)
    plt.xlabel('Dimension 1', fontsize=10)
    plt.ylabel('Dimension 2', fontsize=10)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_original_features.png'), dpi=300)
    plt.close()

def plot_fusion_distribution(source_coords, target_coords, source_labels, target_labels, save_dir, prefix, colors, n_samples=None):
    plt.figure(figsize=(8, 6), dpi=300)
    source_coords = sample_if_needed(source_coords, n_samples)
    target_coords = sample_if_needed(target_coords, n_samples)
    plt.scatter(source_coords[:, 0], source_coords[:, 1], color=colors[0], alpha=0.6, s=15, label='Training Set')
    plt.scatter(target_coords[:, 0], target_coords[:, 1], color=colors[1], marker='x', alpha=0.7, s=15, label='Test Set')
    plt.title(f'Fused Feature Distribution: {prefix}', fontsize=12)
    plt.xlabel('Dimension 1', fontsize=10)
    plt.ylabel('Dimension 2', fontsize=10)
    plt.legend(loc='best')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_fusion_distribution.png'), dpi=300)
    plt.close()

def plot_true_distribution(target_coords, target_labels, class_names, class_mapping, save_dir, prefix, colors, n_samples=None):
    target_coords = sample_if_needed(target_coords, n_samples)
    target_labels = sample_if_needed(target_labels, n_samples)
    plt.figure(figsize=(8, 6), dpi=300)
    unique_classes = np.unique(target_labels)
    for (i, cls) in enumerate(unique_classes):
        cls_name = class_mapping.get(cls, 'Unknown')
        wrapped_name = textwrap.fill(cls_name, width=15)
        idx = target_labels == cls
        color = 'gray' if cls_name == 'Unknown' else colors[i % len(colors)]
        plt.scatter(target_coords[idx, 0], target_coords[idx, 1], color=color, alpha=0.7, s=20, label=wrapped_name)
    plt.title(f'True Label Distribution: {prefix}', fontsize=12)
    plt.xlabel('Dimension 1', fontsize=10)
    plt.ylabel('Dimension 2', fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(unique_classes)), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_true_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_target_pred_distribution(target_coords, pred_target, class_names, class_mapping, private_classes, save_dir, prefix, colors, n_samples=None):
    target_coords = sample_if_needed(target_coords, n_samples)
    pred_target = sample_if_needed(pred_target, n_samples)
    plt.figure(figsize=(8, 6), dpi=300)
    unique_classes = np.unique(pred_target)
    for (i, cls) in enumerate(unique_classes):
        if cls in class_mapping:
            if cls in private_classes:
                cls_name = f'Private: {class_mapping[cls]}'
                marker = 'd'
            else:
                cls_name = class_mapping[cls]
                marker = 'o'
        else:
            cls_name = 'Unknown'
            marker = '*'
        wrapped_name = textwrap.fill(cls_name, width=15)
        idx = pred_target == cls
        if cls in private_classes:
            color = 'purple'
        else:
            color = colors[i % len(colors)]
        plt.scatter(target_coords[idx, 0], target_coords[idx, 1], color=color, alpha=0.7, s=20, marker=marker, label=wrapped_name)
    plt.title(f'Predicted Label Distribution: {prefix}', fontsize=12)
    plt.xlabel('Dimension 1', fontsize=10)
    plt.ylabel('Dimension 2', fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(unique_classes)), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_target_pred_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()

def plot_confusion_matrix(true_labels, pred_labels, class_names, class_mapping, save_dir, prefix):
    """Confusion matrix plot (always full data)"""
    all_classes = np.unique(np.concatenate([true_labels, pred_labels]))
    class_names = [class_mapping.get(cls, f'Class_{cls}') for cls in all_classes]
    cm = confusion_matrix(true_labels, pred_labels, labels=all_classes)
    plt.figure(figsize=(10, 8), dpi=300)
    with np.errstate(divide='ignore', invalid='ignore'):
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent = np.nan_to_num(cm_percent)
    sns.heatmap(cm_percent, annot=cm, fmt='d', annot_kws={'fontsize': 8}, cmap='Blues', xticklabels=class_names, yticklabels=class_names, cbar_kws={'label': 'Percentage'})
    plt.title(f'Confusion Matrix: {prefix}', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=10)
    plt.ylabel('True Class', fontsize=10)
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_confusion_matrix.png'), dpi=300)
    plt.close()

def plot_sankey_diagram(true_labels, pred_labels, class_names, private_classes, save_dir, prefix, colors, max_samples=1000):
    """
    Sankey diagram with compact labels:
    - Left/Right labels use abbreviated names (C0, C1, ...)
    - Legend shows full class names with wrapping
    """
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from pySankey.sankey import sankey
        if len(true_labels) > max_samples:
            indices = np.random.choice(len(true_labels), max_samples, replace=False)
            true_labels = true_labels[indices]
            pred_labels = pred_labels[indices]
        abbr_mapping = {}
        full_name_mapping = {}
        for (i, name) in enumerate(class_names):
            abbr_mapping[i] = f'C{i}'
            full_name_mapping[i] = textwrap.fill(name, width=15)
        for (i, cls) in enumerate(private_classes):
            abbr_mapping[cls] = f'P{i}'
            full_name_mapping[cls] = f'Private Class {cls}'
        if -1 in true_labels or -1 in pred_labels:
            abbr_mapping[-1] = 'Unknown'
            full_name_mapping[-1] = 'Unknown'
        source_labels_abbr = [f'True: {abbr_mapping[label]}' for label in true_labels]
        target_labels_abbr = [f'Pred: {abbr_mapping[label]}' for label in pred_labels]
        df = pd.DataFrame({'Source': source_labels_abbr, 'Target': target_labels_abbr})
        all_classes = sorted(set(true_labels) | set(pred_labels))
        color_mapping = {}
        for (i, cls) in enumerate(all_classes):
            color_mapping[cls] = colors[i % len(colors)]
        color_dict = {}
        for cls in all_classes:
            true_label = f'True: {abbr_mapping[cls]}'
            pred_label = f'Pred: {abbr_mapping[cls]}'
            color_dict[true_label] = color_mapping[cls]
            color_dict[pred_label] = color_mapping[cls]
        plt.figure(figsize=(14, 10), dpi=300)
        sankey(df['Source'], df['Target'], aspect=25, colorDict=color_dict, fontsize=10)
        plt.title(f'True vs Predicted Flow: {prefix}', fontsize=14, pad=20)
        legend_handles = []
        for cls in sorted(all_classes):
            if cls in abbr_mapping and cls in full_name_mapping:
                name = f'{abbr_mapping[cls]}: {full_name_mapping[cls]}'
                legend_handles.append(mpatches.Patch(color=color_mapping[cls], label=name))
        if private_classes:
            legend_handles.append(mpatches.Patch(color='purple', label=f'Private Classes (Count: {len(private_classes)})'))
        plt.legend(handles=legend_handles, loc='lower center', bbox_to_anchor=(0.5, -0.2), ncol=3, fontsize=8, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{prefix}_sankey_diagram.png'), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved Sankey diagram to {os.path.join(save_dir, f'{prefix}_sankey_diagram.png')}")
    except ImportError:
        print("pySankey library not installed. Please install with 'pip install pySankey'")
    except Exception as e:
        print(f'Error generating Sankey diagram: {e}')

def plot_final_classifier_distribution(features, labels, class_names, class_mapping, private_classes, save_dir, prefix, colors, n_samples=None):
    """
    Final classifier distribution plot
    
    Parameters:
        features: feature matrix [N, D]
        labels: predicted labels [N]
        class_names: list of class names
        class_mapping: label to class name mapping
        private_classes: list of private classes
        save_dir: output directory
        prefix: filename prefix
        colors: color list
        n_samples: number of samples (None for full data)
    """
    if n_samples is not None and len(features) > n_samples:
        indices = np.random.choice(len(features), n_samples, replace=False)
        features = features[indices]
        labels = labels[indices]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_jobs=-1)
    embeddings = tsne.fit_transform(features)
    plt.figure(figsize=(10, 8), dpi=300)
    unique_classes = np.unique(labels)
    for (i, cls) in enumerate(unique_classes):
        if cls in class_mapping:
            if cls in private_classes:
                cls_name = f'Private: {class_mapping[cls]}'
                marker = 'd'
                color = 'purple'
            else:
                cls_name = class_mapping[cls]
                marker = 'o'
                color = colors[i % len(colors)]
        else:
            cls_name = 'Unknown'
            marker = '*'
            color = 'gray'
        wrapped_name = textwrap.fill(cls_name, width=15)
        idx = labels == cls
        plt.scatter(embeddings[idx, 0], embeddings[idx, 1], color=color, alpha=0.7, s=20, marker=marker, label=wrapped_name)
    plt.title(f'Final Classifier Distribution: {prefix}', fontsize=12)
    plt.xlabel('t-SNE Dimension 1', fontsize=10)
    plt.ylabel('t-SNE Dimension 2', fontsize=10)
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(4, len(unique_classes)), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{prefix}_final_classifier_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
