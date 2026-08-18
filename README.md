# Clothing Price Predictor — Streamlit

A Streamlit web app that estimates a clothing item's initial listing price
from its attributes, using a Random Forest model trained on real Myntra
catalog data.

## Files

- `app.py` — Streamlit UI and prediction logic.
- `train_model.py` — retrains the model and saves it + the encoder.
- `data_utils.py` — shared data loading/cleaning logic used by both.
- `requirements.txt` — Python packages required.
- `Clothing_dataset.csv` — training data.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The first run automatically trains and saves `Price_Quota_Model.pkl` and
`price_encoder.pkl`. Delete those two files any time to force a retrain
(e.g. after changing the dataset).

## Deploy for free (Streamlit Community Cloud)

1. Push this folder to a GitHub repo.
2. Go to [share.streamlit.io](https://share.streamlit.io), sign in with GitHub.
3. Click **New app**, pick the repo/branch, set the main file to `app.py`.
4. Click **Deploy** — you'll get a public `*.streamlit.app` link in a couple
   of minutes.

## What was fixed from the original version

1. **Missing encoder in deployment.** The original notebook saved the model
   but not the `OneHotEncoder`, so new user inputs at prediction time
   couldn't be transformed consistently. Both are now saved and loaded
   together.
2. **Dataset file was mislabeled.** `Clothing_dataset.csv` was actually an
   Excel file saved with a `.csv` extension, which crashed `pd.read_csv`.
   `data_utils.load_dataset()` now sniffs the real format and falls back to
   `pd.read_excel` automatically, and the shipped file has been converted to
   a genuine CSV.
3. **Corrupted tail rows.** The last several rows of the export have shifted
   columns (they contain data from a different, non-clothing product batch —
   backpacks, belts, etc. — apparently mixed in during the original export),
   which showed up as long free-text/JSON values in the `gender`,
   `occasion`, and `length` columns instead of short category labels. The
   old code cut the dataset at a hard-coded row count (346) that still let
   2 corrupted rows through. Cleaning now detects and drops these rows
   directly by checking those columns for unexpectedly long values, which
   is more robust if the dataset is ever refreshed. This yields 344 clean
   training rows.
4. **UI improvements.** Two-column layout, a sidebar with model info and the
   training price range, and a rough prediction range (min/max across the
   forest's individual trees) alongside the point estimate.

## Model details

- **Algorithm:** RandomForestRegressor (100 trees)
- **Features:** category, gender, brand, fabric, pattern, occasion, length
  (one-hot encoded)
- **Target:** `initial_price` (INR)
- **Training rows:** 344 (after removing corrupted records)

## Limitations

- Trained on ~344 rows from one catalog snapshot — treat predictions as a
  rough estimate, not a precise valuation.
- Brand and category have many categories relative to the dataset size, so
  the model can overfit to specific brand/category combinations seen in
  training.
