# Diagnosis of poor generalisation - measured evidence

**Scope and honesty.** The user's own earlier checkpoints, histories and logs were NOT available when this was written, so
their cause cannot be confirmed from here. Everything below was measured in one environment: TrashNet, the verified split
(1761/380/380 train/val/test), **96x96 inputs, batch 32, seed 42, one CPU core, one seed per run**.
Low resolution lowers absolute accuracy (the 224 px probe below is higher). With 380 test images the 95% interval is about
+-4 points: differences smaller than that are not evidence. Test data was used only for final evaluation; the fine-tune
learning rate was chosen on validation accuracy.

## Confirmed defects (1-2) and a tuning finding (3), all addressed
| # | Defect | Evidence | Fix |
|---|---|---|---|
| 1 | Silent random-weight fallback when the ImageNet download failed (console warning only) | code inspection; a random-weight backbone scores 49.7% in a linear probe vs 72.1% pretrained (same 96 px setup) | training now STOPS (`PretrainedWeightsError`); `--allow-random-init` / `--no-pretrained` are explicit; `weights_source` recorded in meta.json; checksum-verified GitHub mirror for MobileNetV2 |
| 2 | Training used TensorFlow resize, the app used PIL | mean pixel difference 1.7/255, max 71 | one shared loader; training and app inputs are now bit-identical (tested) |
| 3 | Training steps too short / fine-tune LR too timid | 5+5 epochs: val still rising; LR 1e-5 val 73.9% vs 1e-4 val 78.9% | full schedule + `--fine-tune-lr` (chosen on validation) |

Hypotheses tested and REJECTED: BatchNorm stuck in inference mode (statistics do update); BatchNorm moving-average lag
(recalibrating statistics from training data changed nothing). The collapse of a from-scratch MobileNetV2 to one predicted
class on validation was observed but not mechanistically explained.

## Frozen-feature linear probe (no backprop; hyper-parameter C picked on validation)
| Size | Backbone / preprocessing | Test accuracy |
|---|---|---|
| 96px | pretrained + CORRECT preprocessing | 72.1% |
| 96px | pretrained + wrong: rescale to [0,1] | 73.4% |
| 96px | pretrained + wrong: raw 0-255 | 64.5% |
| 96px | pretrained + wrong: ResNet BGR mean-subtract | 65.3% |
| 96px | random weights + correct preprocessing | 49.7% |
| 224px | pretrained + CORRECT preprocessing | 82.9% |

Without feature standardisation (mirrors the real head), 96 px:

| Backbone / preprocessing | Test accuracy |
|---|---|
| pretrained + CORRECT preprocessing | 72.6% |
| pretrained + wrong: rescale to [0,1] | 71.3% |
| pretrained + wrong: raw 0-255 | 65.8% |
| pretrained + wrong: ResNet BGR mean-subtract | 65.8% |
| random weights + correct preprocessing | 23.4% |

Reading: pretrained weights matter most (~23 points); wrong preprocessing costs 1-7 points in a probe.

## Controlled runs (same split, MobileNetV2 unless stated; recall % = plastic / paper / cardboard / glass / metal / general waste)
| run | model | test accuracy (95% CI) | macro F1 | weighted F1 | per-class recall |
|---|---|---|---|---|---|
| repro | Custom CNN | 32.6% (28.1-37.5) | 0.163 | 0.205 | 64 / 87 / 2 / 0 / 0 / 0 |
| repro | MobileNetV2 | 23.4% (19.4-27.9) | 0.063 | 0.089 | 0 / 100 / 0 / 0 / 0 / 0 |
| improved_short | MobileNetV2 | 68.2% (63.3-72.6) | 0.660 | 0.686 | 60 / 69 / 85 / 59 / 74 / 62 |
| improved_long | MobileNetV2 | 70.8% (66.0-75.1) | 0.674 | 0.712 | 69 / 76 / 84 / 57 / 76 / 48 |
| improved_long_ft1e4 | MobileNetV2 | 78.4% (74.0-82.3) | 0.749 | 0.783 | 72 / 89 / 84 / 77 / 77 / 48 |

`repro` = the old silent-fallback behaviour (random weights; MobileNetV2 6 epochs, CNN 5 epochs) reproduced here - NOT the user's checkpoints.
The from-scratch CNN shows the pattern described in the report: training accuracy rising (39% to 51%) while validation
fluctuates (19-31%) with high validation loss.

## Diagnostics of the best run (`improved_long_ft1e4`)
- Label check: loader indices, predictor index-to-name, metadata and output size all consistent = **True**.
- Clean (inference-mode) accuracy: train 94.3%, validation 78.9%, test 78.4% -> gap 15.4 points: **over-fitting suspected** now (flags: overfitting_suspected).
- Weakest classes by recall: general_waste, plastic, glass. Top confusions: {'glass -> metal': 11, 'plastic -> glass': 9, 'general_waste -> paper': 7, 'cardboard -> paper': 6, 'metal -> glass': 6, 'glass -> plastic': 5, 'plastic -> metal': 5, 'paper -> cardboard': 4}.
- Synthetic perturbations (NOT real-world accuracy):

| Probe | Test accuracy | Change (points) |
|---|---|---|
| clean | 78.4% | +0.0 |
| center_only (border grayed out) | 27.9% | -50.5 |
| border_only (object area grayed out) | 35.5% | -42.9 |
| brightness x0.6 | 71.8% | -6.6 |
| brightness x1.4 | 74.2% | -4.2 |
| low contrast (x0.5 around mean) | 71.8% | -6.6 |
| gaussian blur | 70.0% | -8.4 |
| gaussian noise (sigma 15) | 61.1% | -17.4 |
| grayscale (colour removed) | 70.8% | -7.6 |
| rotate 90 deg | 76.8% | -1.6 |
| zoom: centre crop 70% | 76.8% | -1.6 |

The two masking probes create images the model has never seen, so they cannot prove or exclude a background shortcut.
Brightness, rotation and zoom matter little; noise and blur hurt. TrashNet photos are single items on a plain background, so
real-world performance is expected to be lower than these numbers: it has NOT been measured (no real-world photos exist here).

## Not done / limits
- Real-world evaluation (`python -m training.evaluate_real_world`) needs your own labelled photos in `real_world/`.
- ResNet50 was not trained (CPU cost) and has no GitHub weight fallback (legacy file is incompatible).
- Single seed, 96 px; retrain at 224 px on Colab for final numbers.
- Photometric augmentation (`--photometric-aug`) is implemented and tested but its benefit was NOT measured; the diagnostics did not show brightness/contrast as a weakness.
- Inference times depend on the machine; those in the CSV were measured on an otherwise idle single CPU core.

## Reproduce
```powershell
python -m training.prepare_dataset --download
python -m training.probe_features
python -m training.train_mobilenet --tag improved --image-size 224 --fine-tune-lr 1e-4     # Colab GPU recommended
python -m training.evaluate_models --tag improved
python -m training.diagnose --tag improved
python scripts/compare_runs.py --tags original improved
```
