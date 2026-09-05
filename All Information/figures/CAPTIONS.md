# Figure manifest — model, split, and source for every image
(Caption mislabeling — e.g. presenting a validation figure as test — is a
classic examiner catch; this manifest is the ground truth.)

| figure | model | split | shows | source (repo) |
|---|---|---|---|---|
| fig_det_curves_yolo26m.png | yolo26m·det11 | train/val | 200-epoch loss + mAP curves | reports/figures/yolo26m_11/results.png |
| fig_det_curves_yolo12m.png | yolo12m·det11 | train/val | 200-epoch loss + mAP curves | reports/figures/yolo12m_11/results.png |
| fig_det_cm_val_yolo26m.png | yolo26m·det11 | **val** | normalized confusion matrix | reports/figures/yolo26m_11/confusion_matrix_normalized.png |
| fig_det_pr_val_yolo26m.png | yolo26m·det11 | **val** | precision-recall curves | reports/figures/yolo26m_11/BoxPR_curve.png |
| fig_det_predictions_val.png | yolo26m·det11 | **val** | prediction mosaic | reports/figures/yolo26m_11/val_batch0_pred.jpg |
| fig_det22_curves_yolo26n.png | yolo26n·det22 | train/val | training curves (22-class) | reports/figures/yolo26n_22/results.png |
| fig_det22_cm_val_yolo26n.png | yolo26n·det22 | **val** | normalized confusion matrix (22-class) | reports/figures/yolo26n_22/confusion_matrix_normalized.png |
| fig_cls_curves_swin.png | swin | train/val | 30-epoch loss + F1/acc curves | reports/figures/cls_swin/results.png |
| fig_cls_cm_val_swin.png | swin | **val** | binary confusion matrix | reports/figures/cls_swin/confusion_matrix.png |
| fig_cls_curves_efficientnet.png | efficientnet_b0 | train/val | training curves | reports/figures/cls_efficientnet/results.png |
| fig_cls_curves_dinov2.png | dinov2-s | train/val | training curves | reports/figures/cls_dinov2/results.png |
| fig_test_cm_yolo26m.png | yolo26m·det11 | **TEST** | normalized confusion matrix | reports/figures/test_yolo26m_det11/confusion_matrix_normalized.png |
| fig_test_pr_yolo26m.png | yolo26m·det11 | **TEST** | precision-recall curves | reports/figures/test_yolo26m_det11/BoxPR_curve.png |
| fig_test_predictions_yolo26m.png | yolo26m·det11 | **TEST** | prediction mosaic | reports/figures/test_yolo26m_det11/val_batch0_pred.jpg |
| fig_test_groundtruth_yolo26m.png | (ground truth) | **TEST** | annotated ground-truth mosaic — pair with fig_test_predictions_yolo26m for a GT-vs-prediction figure | runs_evaluation/yolo26m_det11_test/val_batch0_labels.jpg |
| fig_test_cm_swin.png | swin | **TEST** | binary confusion matrix (2,233 crops) | reports/figures/test_cls_swin/confusion_matrix.png |
| fig_test_cm_yolo26n22.png | yolo26n·det22 | **TEST** | normalized confusion matrix (22-class) | reports/figures/test_yolo26n_det22/confusion_matrix_normalized.png |
| fig_test_f1_curve_yolo26m.png | yolo26m·det11 | **TEST** | F1 vs confidence threshold (operating-point choice) | runs_evaluation/yolo26m_det11_test/BoxF1_curve.png |
| fig_dataset_stats.png | (dataset) | train | class-frequency histogram + bbox size/position densities | runs_detection/yolo26m_det11/labels.jpg |
| fig_augmentation_batch.png | (pipeline) | train | augmented training batch (mosaic etc.) | runs_detection/yolo26m_det11/train_batch0.jpg |
