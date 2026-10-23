# Demonstration Checklist and Viva Q&A

## Demo checklist (about 8 minutes)
- [ ] Before the demo: trained models exist, `python -m training.evaluate_models` has been run, `python -m pytest -q` passes.
- [ ] `streamlit run app.py`; browser at localhost:8501; camera permission pre-granted; 3-4 test photos ready (include one hard/cluttered one).
- [ ] Dashboard: explain cards; show empty state if history is empty.
- [ ] Classify Waste: upload -> select model -> classify -> confidence, top-3, advice, bin simulation.
- [ ] Toggle Grad-CAM; explain it shows model focus, not correctness.
- [ ] Switch model and compare results on the same image.
- [ ] Live Detection: capture -> classify (have the upload fallback ready).
- [ ] Model Performance: comparison table, confusion matrix, which class is weakest and why.
- [ ] History: filter, CSV export. Dashboard now populated.
- [ ] Show an invalid file (e.g. .txt) to demonstrate error handling.
- [ ] State limitations honestly (small dataset, white backgrounds, simulation only).

## Viva questions (answer with your OWN measured numbers)
1. **Why deep learning instead of classical ML?** CNNs learn features (edges -> textures -> shapes) directly from pixels; hand-crafted features generalise worse on varied images.
2. **Explain your custom CNN.** Four blocks of Conv-BN-ReLU twice then MaxPool; filters 32->256; global average pooling; dense layer with dropout; softmax over 6 classes.
3. **What is transfer learning and why use it here?** Reuse ImageNet features; the dataset is small (~2.5k), so training from scratch overfits.
4. **Why freeze the backbone first?** The new random head would send large gradients into good pretrained weights; train the head, then fine-tune gently with a low learning rate.
5. **Why keep BatchNorm frozen when fine-tuning?** Small batches give noisy statistics; frozen BN keeps pretrained statistics stable.
6. **Why different preprocessing per model?** Each backbone was trained with specific scaling (MobileNetV2: [-1,1]; ResNet50: BGR mean-subtract). Using the same function in training and inference avoids mismatch.
7. **How did you prevent data leakage?** Test set held out; exact duplicates removed; byte-identical images with different labels removed; near-duplicate photos (16x16 thumbnail difference, threshold chosen by inspecting real pairs; an 8x8 dHash was rejected for false positives) grouped so they cannot straddle splits; split verified and hashed; seed fixed. Known gap: same object from a different pose is not detected.
8. **How did you handle class imbalance?** Stratified splits plus class weights in the loss; macro-F1 reported, not only accuracy.
9. **Why macro-F1 as well as accuracy?** Accuracy hides poor performance on the small `general_waste` class.
10. **What does the confusion matrix show?** Rows = true class, columns = predicted; off-diagonal cells are the confusions (read yours from `results/`).
11. **How did you measure inference time?** Median of 100 single-image passes after warm-up, model only (documented in `evaluate_models.py`).
12. **How did you pick the deployment model?** Rule in code: highest macro-F1, ties within 0.01 go to the fastest.
13. **What is Grad-CAM?** Weights the last conv feature maps by the gradient of the class score; ReLU; upsample; overlay. Highlights influential regions; does not prove correctness.
14. **Is confidence the probability of being correct?** No: it is the softmax output, and can be overconfident, especially out of distribution.
15. **What would break in the real world?** Clutter, multiple items, lighting; the dataset has white backgrounds. Mitigation: more diverse data, augmentation, object detection.
16. **Why map `trash` to `general_waste`?** Closest semantic match; documented in `config.yaml`; it is the smallest, noisiest class.
17. **Is the smart bin real?** No - it is a software simulation; a servo-based sorter is future work.
18. **What are the privacy considerations?** Images are not saved by default; only filename and prediction are stored locally.
19. **How do you know the app works?** Automated tests (preprocessing, DB, predictor, Grad-CAM, split leakage, page smoke tests) and manual end-to-end run.
20. **What would you improve first?** Collect real-world images, calibrate confidence, add detection for multiple objects.
