# Real-world evaluation images (yours - none are included)

Put photos you took yourself in the folder of their TRUE category: `plastic/ paper/ cardboard/ glass/ metal/ general_waste/`.
Photos without a known label go in `unlabeled/` (you get predictions and confidence, never an accuracy number).

Aim for at least 10 images per class (30+ is better), deliberately varied:
different backgrounds (table, floor, bin, hand), lighting (daylight, dim, flash), angles and distances,
crumpled/folded/dirty/partly hidden items, and several different physical objects per class.
Do NOT photograph the same object many times and split the copies across sets.

Optional `conditions.csv` (header `file,condition`, file relative to this folder, e.g. `plastic/p1.jpg,dim light`)
gives accuracy per condition.

```
python -m training.evaluate_real_world --model mobilenetv2 --tag improved
```
Keep this set as a FINAL test: after ~3 evaluations the script warns that it is no longer independent. If you later
use these photos to fine-tune the model, collect a new set for testing.
