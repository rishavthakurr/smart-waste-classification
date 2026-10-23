# Application Guide

Start the app with the command `streamlit run app.py` (opens http://localhost:8501). The app only **loads** trained models;
it never trains and never substitutes one model for another.

| Page | Controls and behaviour |
|---|---|
| **Dashboard** | Cards (images classified, classes, most-predicted category, deployment model); **Models** panel (Ready / Smoke test only / Not trained + the exact training command); donut of category shares; grouped bars by category and model; daily activity; last 10 predictions; model comparison chart. All values come from the SQLite database and `results/`. Empty states explain how to create the first prediction. |
| **Classify Waste** | Upload JPG/PNG (validated: type, size, decodability, dimensions) -> choose model (radio; untrained models are listed with their training command) -> **Classify Waste** -> predicted category, real softmax confidence, probability chart, top-3, disposal advice, smart-bin simulation, optional **Grad-CAM**. **Classify another image** resets. Every classification is stored in SQLite. |
| **Live Detection** | Webcam capture (browser permission needed; `localhost`/HTTPS) -> classify; **Capture another image**; **Upload instead** tab as a fallback. |
| **Model Performance** | Reads `results/`: comparison table and charts, per-class metrics, confusion matrix, training curves, dataset report. Shows "pending" with commands if nothing has been evaluated. |
| **Classification History** | Filter by category and model, filename search, sort by time, summary metrics, **Download CSV** (exports exactly the filtered rows), **Clear history** (checkbox-confirmed). |
| **About** | Scope, class mapping, dataset attribution, limitations. |

## Notices you may see
- *Smoke-test model*: trained only to verify the pipeline; its predictions are meaningless.
- *Without ImageNet weights*: a transfer model trained from random initialisation.
- *Low confidence*: top probability below `ui.low_confidence_threshold` (config.yaml).

## Error handling
Invalid/corrupt/oversized uploads, missing model files, corrupted model files, class-order mismatches, inference failures and
Grad-CAM failures all produce a plain-language message; stack traces go to the server log only.

## Testing the app
```powershell
python -m pytest -q                              # unit, pipeline and headless page tests
pip install -r requirements-dev.txt; python -m playwright install chromium
python -m pytest tests/test_browser_e2e.py -q    # real browser: upload -> classify -> history -> CSV
```
