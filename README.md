# Smart Waste Classification and Intelligent Segregation System Using Deep Learning

A final-year MCA major project: classify waste photos into six categories with deep learning, explain
predictions with Grad-CAM, recommend disposal actions, and track everything in a Streamlit dashboard.

> **Honesty note.** This repository ships **no trained models and no accuracy numbers.** Every metric shown
> by the app is computed by `training/evaluate_models.py` on *your* trained models. Until you train, the app
> says "training and evaluation pending".

## 1-2. Abstract and problem statement
Manual waste segregation is slow, inconsistent and error-prone, and contaminated recycling streams lower
recycling yields. This project builds an image-based classifier (custom CNN, MobileNetV2, ResNet50),
compares the three on one held-out test set, and wraps the best into a web app that predicts the material,
shows confidence, explains the decision, and suggests a disposal action.

## 3. Objectives
1. Build a leakage-safe dataset pipeline (validation, de-duplication, stratified split).
2. Train and compare a custom CNN with two transfer-learning models.
3. Evaluate with accuracy, precision/recall/F1 (per-class, macro, weighted), confusion matrices, speed and size.
4. Provide an interactive app: upload / webcam classification, history, analytics, Grad-CAM.
5. Stay reproducible and honest: fixed seeds, no fabricated results.

## 4. Methodology
Dataset (TrashNet) -> `prepare_dataset.py` (map labels, drop corrupt/duplicate files, group near-duplicates,
70/15/15 split) -> training (augmentation, class weights, early stopping, checkpointing; transfer models:
train head, then cautiously fine-tune last backbone layers with BatchNorm frozen) -> `evaluate_models.py`
(same test set for all models) -> Streamlit app loads the saved model; it never trains.

## 5. System architecture
```
config.yaml ──────────────┬─────────────────────────────────────────────┐
                          ▼                                             ▼
 dataset/raw ─► prepare_dataset ─► dataset/{train,validation,test} ─► train_*.py ─► models/*.keras
                                             │                                  └► results/<model>/
                                             └──────────► evaluate_models.py ──► results/*.csv, *.png
 Streamlit app (app.py + pages/)
   ui.py ─► predictor.py ─► model_loader.py ─► models/*.keras        gradcam.py
        └─► database.py (SQLite)      recommendations.py (guidance + bin simulation)
```

## 6-7. Requirements
**Hardware:** any laptop for the app and for small CPU experiments (4 GB RAM minimum); a GPU (e.g. free Google
Colab T4) is strongly recommended for full training. **Software:** Python 3.11 (3.10-3.12 should work),
the packages in `requirements.txt`, a modern browser (webcam needs `localhost` or HTTPS).

## 8. Dataset setup
Dataset: **TrashNet** (Thung & Yang, Stanford CS229, 2016): 2,527 images, 6 folders: cardboard, glass, metal,
paper, plastic, trash. Licence: **MIT** (please cite the repository). Source: <https://github.com/garythung/trashnet>.

Label mapping (documented in `config.yaml`): plastic, paper, cardboard, glass, metal map 1:1; TrashNet
**`trash` -> `general_waste`**. 

Automatic: `python -m training.prepare_dataset --download`. Manual: download `dataset-resized.zip` from the repo's
`data/` folder, extract it so that `dataset/raw/dataset-resized/<class>/` exists, then run
`python -m training.prepare_dataset`. The script writes `dataset/{train,validation,test}/<class>/`,
`dataset/manifest.csv`, `results/dataset_report.json`, `results/dataset_excluded_images.csv` and charts in `results/dataset/`.

The script first **inspects** the real folder structure (`--inspect` shows it without writing anything) and aborts if
any project class has no data, so a six-output model can never be trained on five classes. It then drops corrupt files,
byte-identical duplicates and identical images filed under conflicting labels (TrashNet contains 3 such pairs), groups
near-duplicate photos (16x16 thumbnail difference, threshold chosen by inspecting real pairs) so they stay in one split,
splits 70/15/15 with a fixed seed, rebuilds the split folders from scratch, and verifies the result (files on disk equal the
manifest; no cluster or identical image crosses splits). Re-running gives a byte-identical `manifest.csv`.
Measured on TrashNet: 2,521 usable images (6 excluded), split 1761 / 380 / 380, 5 near-duplicate groups (19 images).
**Limitation:** no cheap image hash detects the same physical object photographed from a different pose, so test
accuracy on TrashNet may be somewhat optimistic.

## 9. Installation (Windows PowerShell)
```powershell
cd smart-waste-classification
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
python -m pip install --upgrade pip
pip install -r requirements.txt
python scripts/setup_project.py --download-data
```
Linux/macOS: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.

## 10. Training (each model is independent; full guide in `docs/TRAINING.md`)
```powershell
python -m training.train_cnn                      # Model A: custom CNN
python -m training.train_mobilenet                # Model B: MobileNetV2 (downloads ImageNet weights, needs internet)
python -m training.train_resnet                   # Model C: ResNet50 (larger; GPU recommended)
```
Useful flags: `--epochs 10`, `--batch-size 16`, `--image-size 128`, `--cpu`, `--no-fine-tune`, `--no-pretrained`,
and **`--smoke`** (tiny 2-epoch run to verify the pipeline - its output is *not* a result). Defaults live in `config.yaml`.
If ImageNet weights cannot be downloaded the script warns loudly, falls back to random initialisation and records
`pretrained_imagenet: false`; the app shows a notice for such models.

### Google Colab (GPU)
Easiest: open `notebooks/colab_training.ipynb` in Colab. Manual steps:
1. Runtime -> Change runtime type -> GPU. Upload the project folder (zip) or `git clone` your repo.
2. ```
   !pip install -r requirements-colab.txt
   %cd smart-waste-classification
   !python -m training.prepare_dataset --download
   !python -m training.train_mobilenet
   !python -m training.train_cnn
   !python -m training.train_resnet
   !python -m training.evaluate_models
   !zip -r outputs.zip models results
   ```
3. Download `outputs.zip`, extract it into your local project (keep the `models/` and `results/` folders).

## 11. Evaluation
```powershell
python -m training.evaluate_models                # all trained models
python -m training.evaluate_models --models cnn mobilenetv2
```
Outputs in `results/`: `model_comparison.csv`, `per_class_metrics.csv`, `model_comparison.png`, `per_class_f1.png`,
`recommended_model.json`, and per model `confusion_matrix.{csv,png}`, `history.csv`, `training_curves.png`, `meta.json`.
Inference time = median of 100 single-image forward passes after 10 warm-ups (model only; excludes image loading).
Recommendation rule: highest macro-F1; among models within 0.01 of the best, the fastest.

## 12. Run the app
```powershell
streamlit run app.py
```
Opens at <http://localhost:8501>. The app only *loads* trained models. Models are discovered from `models/`.

## 13. Features (all implemented)
| Page | What it does |
|---|---|
| Dashboard | Totals, most-predicted class, deployment model, category charts, activity, recent predictions, model comparison (real stored data only; friendly empty state) |
| Classify Waste | JPG/PNG upload + validation, model selector, prediction, true softmax confidence, probability chart, top-3, disposal advice, bin simulation, Grad-CAM, "classify another" |
| Live Detection | Webcam capture -> classify; upload fallback; capture-another |
| Model Performance | Reads real evaluation files: comparison table, per-class metrics, confusion matrices, training curves, dataset report |
| Classification History | SQLite-backed; filter by category/model, filename search, sort, CSV export, confirmed clear |
| About | Scope, class mapping, dataset attribution, limitations |

Unavailable models are listed with the command to train them and are never silently substituted.
Images are **not** stored (config `ui.store_uploaded_images` is false); only filename, prediction and probabilities are saved.

## 14. Project structure
```
app.py  config.yaml  requirements*.txt  pytest.ini
training/  prepare_dataset.py  models.py  common.py  train_{cnn,mobilenet,resnet}.py  evaluate_models.py
src/       utils.py preprocessing.py model_loader.py predictor.py recommendations.py gradcam.py database.py ui.py
pages/     dashboard.py classify_waste.py live_detection.py model_performance.py history.py about.py
tests/     test_preprocessing.py test_predictor.py test_database.py test_recommendations.py
           test_training_components.py test_dataset_and_parity.py test_pipeline_e2e.py test_app_smoke.py test_browser_e2e.py
scripts/   setup_project.py        notebooks/ colab_training.ipynb
docs/      REPORT_OUTLINE.md  DEMO_AND_VIVA.md  TRAINING.md
models/ dataset/ results/ assets/   (generated; git-ignored)
```

## 15. Expected outputs
After the steps above: trained `.keras` files in `models/`, metrics/plots in `results/`, a working app,
and `results/classification_history.db` created on first use.

## 16. Testing
```powershell
python -m pytest -q
```
Page-by-page behaviour is in `docs/APP_GUIDE.md`. The optional real-browser test needs `pip install -r requirements-dev.txt` and `python -m playwright install chromium`.
Unit tests use mocked model outputs or tiny randomly-initialised models to test *mechanics* only (never reported as results).
`test_app_smoke.py` renders every page headlessly. Manual smoke test: `streamlit run app.py`, open each page, classify one image.

## 16b. If the models generalise poorly
Read `docs/GENERALIZATION_DIAGNOSIS.md` (measured evidence, confirmed defects, and the commands to diagnose your own runs).
Key points: transfer training now **stops** if ImageNet weights cannot be loaded (no silent random-weight fallback);
`python -m training.diagnose`, `training.evaluate_real_world` (your own photos in `real_world/`), `scripts/compare_runs.py`
and `scripts/promote_model.py` support a before/after comparison; runs are kept apart with `--tag`.
`models/improved_long_ft1e4/` is a 96 px, CPU-trained experiment (see the report), not a production model.

## 17. Known limitations
- Small dataset (~2.5k images), single item on a white background: expect lower accuracy on cluttered real photos.
- `general_waste` is the smallest class (~5%); class weights help but do not remove imbalance.
- One item per image; no object detection. Disposal advice is generic and depends on local rules.
- Softmax confidence is not a calibrated probability of being correct. Grad-CAM shows model focus, not correctness.
- Webcam capture depends on browser permissions and HTTPS/localhost.
- Fine-tuning code runs only with pretrained weights; it needs an internet connection to fetch them.

## 18. Future enhancements (not implemented)
Physical sorter (microcontroller + servo) driven by the predicted class; continuous video mode; multi-object detection
(YOLO); larger/more diverse dataset and calibration; mobile deployment (TFLite); per-city recycling rules.

## 19. Attribution
TrashNet - Gary Thung and Mindy Yang, 2016, MIT licence, <https://github.com/garythung/trashnet>. Pretrained backbones:
ImageNet weights distributed with Keras Applications (MobileNetV2: Sandler et al. 2018; ResNet50: He et al. 2016).
Grad-CAM: Selvaraju et al. 2017.
