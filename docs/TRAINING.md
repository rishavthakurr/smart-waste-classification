# Training and Evaluation Guide

All commands run from the project root. Defaults come from `config.yaml`; flags override them.

## 0. Pipeline at a glance
```
prepare_dataset  ->  dataset/{train,validation,test}/ + dataset/manifest.csv + results/dataset_*.json|csv|png
train_<model>    ->  models/<model>.keras + results/<model>/{history.csv, meta.json, training_curves.png}
evaluate_models  ->  results/{model_comparison.csv, per_class_metrics.csv, evaluation_summary.json,
                              recommended_model.json, *.png} + results/<model>/{metrics.json,
                              confusion_matrix.csv|png, test_predictions.csv}
streamlit run app.py   (reads the files above; never trains)
```

## 1. Dataset
```powershell
python -m training.prepare_dataset --inspect     # look at the real folders only (writes nothing)
python -m training.prepare_dataset --download    # download TrashNet, inspect, prepare, verify
python -m training.prepare_dataset               # prepare images already in dataset/raw
```
Labels are never assumed: the folders on disk are listed, the mapping in `config.yaml` is checked against them,
and the run aborts if any of the six classes has fewer than `dataset.min_images_per_class` usable images.

## 2. Smoke test on CPU (about 1-2 minutes per model; verifies the code, results are meaningless)
```powershell
python -m training.train_cnn --smoke --cpu
python -m training.train_mobilenet --smoke --cpu --no-pretrained --debug-finetune-random-init
python -m training.evaluate_models
```
Smoke models are marked `smoke_test: true` in `meta.json`; the app shows warnings for them. **Delete `models/` and `results/<model>/`
before real training** (or just retrain: files are overwritten).

## 3. Real training (Colab GPU recommended; CPU works but is slow)
```powershell
python -m training.train_cnn
python -m training.train_mobilenet
python -m training.train_resnet
python -m training.evaluate_models
```
| Flag | Meaning |
|---|---|
| `--epochs N`, `--batch-size N`, `--lr X`, `--image-size N` | override config defaults |
| `--no-fine-tune` | train only the new head for transfer models |
| `--no-pretrained` | deliberately train from random weights (not transfer learning) |
| `--allow-random-init` | if ImageNet weights cannot be loaded, continue from random weights instead of STOPPING (default is to stop) |
| `--photometric-aug` | brightness/contrast/saturation jitter on TRAINING images only |
| `--fine-tune-lr X` | fine-tuning learning rate (config default 1e-5) |
| `--tag NAME` | save to `models/NAME/` and `results/NAME/` so runs never overwrite each other |
| `--cpu` | hide GPUs |
| `--deterministic` | TensorFlow op determinism (repeatable, slower on GPU) |
| `--smoke` | tiny pipeline check |
| `--debug-finetune-random-init` | testing only: run the fine-tune phase without pretrained weights |

What each run does: seeds everything; loads images with the same PIL code the app uses; applies the architecture's own scaling
(`rescale_01`, MobileNetV2 `[-1,1]`, ResNet50 BGR mean-subtraction); applies geometric augmentation inside the model (training only);
uses class weights; checkpoints the best `val_accuracy`; early stops; reduces LR on plateau; for transfer models trains the head,
then (only if ImageNet weights loaded) fine-tunes the last 30 backbone layers with BatchNorm frozen at LR 1e-5.

## 4. Output files
| File | Content |
|---|---|
| `results/<m>/history.csv` | per-epoch loss, accuracy, val_loss, val_accuracy, learning_rate, phase |
| `results/<m>/meta.json` | seed, sizes, pretrained?, fine-tuned?, smoke?, best epoch, split sha256, versions, command |
| `results/<m>/training_curves.png` | accuracy and loss plots |
| `results/<m>/metrics.json` | sklearn classification report (per-class P/R/F1, macro, weighted), confusion matrix |
| `results/<m>/test_predictions.csv` | every test image: true, predicted, confidence, all class probabilities |
| `results/model_comparison.csv` | accuracy, macro/weighted P/R/F1, ms per image, MB, parameters, flags |
| `results/evaluation_summary.json` | everything above in one file + timing procedure |
| `results/recommended_model.json` | deployment pick and the rule used |

## 4b. Diagnosing poor generalisation (see docs/GENERALIZATION.md)
```powershell
python -m training.probe_features                 # pretrained vs random features, preprocessing sensitivity (linear probe)
python -m training.diagnose --tag improved        # label check, clean train/val/test accuracy, errors, perturbation probes
python -m training.evaluate_models --tag improved # same test split for every run
python -m training.evaluate_real_world --model mobilenetv2 --tag improved   # your own photos in real_world/
python scripts/compare_runs.py --tags original improved                     # one CSV, with 95% intervals
python scripts/promote_model.py --tag improved --model mobilenetv2          # make it the app's model (original preserved)
```

## 5. Safety checks you get for free
- Evaluation **refuses** models trained on a different split (`manifest_sha256` in `meta.json`); `--force` overrides.
- A model trained with a different class order than `config.yaml` is rejected when loaded.
- Transfer training **stops with an error** if ImageNet weights cannot be loaded (no silent random-weight fallback). Weight source and checksum-verified mirror are recorded in `meta.json` (`weights_source`).
- Models without ImageNet weights, and smoke models, are labelled as such everywhere in the app.

## 6. Troubleshooting
| Symptom | Fix |
|---|---|
| `PretrainedWeightsError` | no route to ImageNet weights: fix network/proxy, or place an official Keras no-top file at `models.<m>.weights_path`; MobileNetV2 also falls back to a checksum-verified GitHub mirror |
| Out-of-memory | lower `--batch-size` or `--image-size`; images are held in RAM as uint8 |
| `DATASET CHECK FAILED` | read the listed class; fix `label_map` or supply the data |
| Evaluation says "different dataset split" | you re-ran `prepare_dataset` after training; retrain |
| Slow on laptop | use `--cpu --smoke` to test, Colab GPU for real runs |
