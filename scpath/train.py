from scpath.utils.data_prepare import prepare_data
import torch
from scpath.fusion_train import train_joint_with_da
from scpath.distill_train import DistillConfig, prepare_distill_data, train_student_model, run_reclamation, evaluate_and_plot, update_and_save_adata
import os

def train(config, adata_train_rna, adata_test_rna, pathway_names, train_fusion=True, distill=True):
    """Run the original joint-training and student-reclamation stages in order."""
    (adata_train, adata_test, label_mapping, knn_graphs) = prepare_data(config.path_output, adata_train_rna, adata_test_rna, pathway_names)
    print(list(label_mapping['cell_type']))
    if train_fusion:
        train_joint_with_da(adata_train, adata_test, knn_graphs, label_mapping, config, pathway_names)
    if distill:
        my_config = DistillConfig(config)
        test_file = os.path.join(config.log_dir, 'test_results.h5ad')
        (data_tensors, meta_info) = prepare_distill_data(my_config, test_file)
        student_model = train_student_model(my_config, data_tensors, meta_info['num_classes'])
        (final_preds, student_preds, reclaimed_mask) = run_reclamation(my_config, student_model, data_tensors, meta_info)
        final_acc = evaluate_and_plot(meta_info, final_preds, student_preds, config.log_dir)
        adata_distilled = update_and_save_adata(my_config, test_file, final_preds, student_preds, reclaimed_mask)
