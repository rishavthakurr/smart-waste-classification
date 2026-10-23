# Major Project Report Outline

Legend: **[Implemented]** exists in this repo · **[Fill in]** you must supply from your own runs · **[Future]** not built.
Do not copy numbers from anywhere except `results/` after YOUR training and evaluation.

1. **Abstract** - problem, approach (CNN + MobileNetV2 + ResNet50, Grad-CAM, Streamlit), dataset, headline result [Fill in from `model_comparison.csv`].
2. **Introduction** - waste problem, role of computer vision.
3. **Problem statement** - classify a waste image into six classes and advise disposal.
4. **Motivation** - recycling contamination, automation, education.
5. **Objectives** - see README section 3.
6. **Literature review outline** - [Fill in] sources you have actually read. Suggested themes: TrashNet and CS229 baseline (Thung & Yang 2016); CNN image classification; transfer learning (MobileNetV2, ResNet); explainability (Grad-CAM, Selvaraju et al. 2017); class imbalance. Verify every citation yourself before listing it.
7. **Existing system** - manual segregation; classical ML on hand-crafted features; limitations.
8. **Proposed system** - [Implemented] architecture diagram in README section 5.
9. **System architecture** - modules: dataset pipeline, training, evaluation, inference, database, UI.
10. **Dataset description** - TrashNet, 2,527 images, MIT licence, label mapping (`trash`->`general_waste`), imbalance, white-background caveat; use `results/dataset_report.json` and `results/dataset/*.png`.
11. **Data preprocessing** - validation, exact-duplicate removal, conflicting-label removal, near-duplicate grouping (thumbnail difference; state the limitation), grouped stratified 70/15/15 split, resizing, model-specific scaling, training-only augmentation.
12. **CNN methodology** - Model A architecture (4 conv blocks, BN, ReLU, pooling, dropout, GAP, dense head).
13. **Transfer-learning methodology** - frozen backbone head training, then fine-tuning last N layers with low LR and frozen BatchNorm; model-specific preprocessing.
14. **Training procedure** - Adam, sparse categorical cross-entropy, class weights, early stopping, ReduceLROnPlateau, checkpoint on val accuracy, seeds.
15. **Experimental setup** - [Fill in] hardware (Colab GPU / laptop), TF version, hyper-parameters (`meta.json`).
16. **Evaluation metrics** - accuracy, precision, recall, F1 (per-class, macro, weighted), confusion matrix, inference time (procedure in `evaluate_models.py`), model size.
17. **Actual results** - [Fill in] tables and figures from `results/`. Report what you measured, including weak classes.
18. **Discussion** - which model won and why (use *measured* accuracy/speed/size); confusion patterns; Grad-CAM examples; effect of pretrained weights.
19. **Limitations** - README section 17; plus any you observe.
20. **Conclusion** - restate measured outcomes.
21. **Future scope** - [Future] items in README section 18.
22. **References** - only sources you verified.
