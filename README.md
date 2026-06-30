# Customer Purchase Prediction System

An end-to-end machine learning system that predicts whether a customer is likely to respond to a marketing campaign (i.e. make a purchase), built on the classic **Marketing Campaign** dataset. The project covers data preprocessing, exploratory data analysis, model training with hyperparameter tuning, a FastAPI inference service, and an interactive Streamlit dashboard.

---

## 1. Project Structure

```
korkai/
├── api/
│   └── main.py                  # FastAPI app for serving predictions
├── data/
│   └── marketing_campaign.csv   # Raw dataset (tab-separated)
├── models/
│   ├── model.pkl                 # Trained best model (full sklearn Pipeline)
│   └── decision_threshold.pkl    # Tuned probability threshold for classification
├── notebook/
│   └── EDA.ipynb                 # Exploratory Data Analysis notebook
├── reports/
│   ├── model_comparison.csv               # Metrics for every model trained
│   ├── feature_importance.csv             # Random Forest feature importances
│   ├── *_confusion_matrix.png             # Confusion matrix per model
│   └── Project_Report.pdf                 # Auto-generated PDF report (created on training)
├── src/
│   ├── preprocessing.py          # Data cleaning + feature engineering
│   ├── train.py                  # Model training, tuning, evaluation, saving
│   └── report_generator.py       # Builds the PDF summary report
├── app.py                        # Streamlit dashboard (EDA + live prediction + batch prediction)
├── requirements.txt
└── README.md
```

---

## 2. Dataset

**File:** `data/marketing_campaign.csv` (tab-separated, ~2,240 rows, 29 raw columns)

This is a customer relationship management / marketing campaign dataset. Key raw columns:

| Column | Description |
|---|---|
| `ID` | Unique customer identifier (dropped during cleaning) |
| `Year_Birth` | Customer's birth year |
| `Education` | Education level |
| `Marital_Status` | Marital status |
| `Income` | Yearly household income |
| `Kidhome` / `Teenhome` | Number of small children / teenagers at home |
| `Dt_Customer` | Date the customer enrolled with the company |
| `Recency` | Days since last purchase |
| `MntWines`, `MntFruits`, `MntMeatProducts`, `MntFishProducts`, `MntSweetProducts`, `MntGoldProds` | Amount spent on each product category (last 2 years) |
| `NumDealsPurchases` | Purchases made with a discount |
| `NumWebPurchases` / `NumCatalogPurchases` / `NumStorePurchases` | Purchases made via each channel |
| `NumWebVisitsMonth` | Website visits in the last month |
| `AcceptedCmp1`-`AcceptedCmp5` | Whether customer accepted the offer in each of 5 prior campaigns |
| `Complain` | Whether customer complained in the last 2 years |
| `Z_CostContact`, `Z_Revenue` | Constant columns (dropped, no predictive value) |
| `Response` | **Target** — 1 if customer accepted the offer in the last campaign, 0 otherwise |

---

## 3. Preprocessing & Feature Engineering (`src/preprocessing.py`)

`preprocess(file_path)` runs three stages:

**`load_data`** — reads the CSV (tab-separated).

**`clean_data`**
- Fills missing `Income` with the median.
- Drops the `Income == 666666` outlier record.
- Caps `Income` at the 99th percentile to limit the effect of extreme outliers.
- Drops unrealistic birth years (`Year_Birth < 1940`).
- Drops non-predictive columns: `ID`, `Z_CostContact`, `Z_Revenue`.
- Collapses rare `Marital_Status` categories (`YOLO`, `Absurd`, `Alone`) into `Other`.

**`feature_engineering`** (uses a fixed reference date = `max(Dt_Customer) + 1 day`, so results stay reproducible regardless of when the script is run)
- `Age` = reference year − `Year_Birth`
- `Customer_Tenure` = days between reference date and `Dt_Customer`
- `Previous_Purchase` = sum of web + catalog + store purchases
- `Total_Spending` = sum of all `Mnt*` product spend columns
- `Children` = `Kidhome` + `Teenhome`
- `TotalAcceptedCmp` = sum of `AcceptedCmp1`…`AcceptedCmp5`
- Drops the now-redundant `Dt_Customer` column

The final feature set used for modeling/inference is:

```
Age, Income, Education, Marital_Status, Children, Customer_Tenure,
Recency, NumDealsPurchases, NumWebVisitsMonth, Previous_Purchase,
Total_Spending, TotalAcceptedCmp, Complain
```
Target: `Response`

You can run preprocessing standalone:
```bash
cd src
python preprocessing.py
```

---

## 4. Exploratory Data Analysis (`notebook/EDA.ipynb`)

Jupyter notebook exploring distributions, correlations, and patterns in the cleaned dataset (income, spending, channel behavior, response rate, etc.). Open it with:
```bash
jupyter notebook notebook/EDA.ipynb
```
A condensed, interactive version of this EDA is also reproduced live in the **Streamlit dashboard** (Tab 2).

---

## 5. Model Training (`src/train.py`)

Trains and compares four classifiers using `GridSearchCV` (5-fold `StratifiedKFold`, scored on F1) inside a `ColumnTransformer` + `Pipeline`:

- **Logistic Regression** (`class_weight="balanced"`)
- **Decision Tree** (`class_weight="balanced"`)
- **Random Forest** (`class_weight="balanced"`)
- **XGBoost** (`scale_pos_weight` set for class imbalance)
- **Voting Ensemble** — soft-voting combination of the tuned Logistic Regression + Random Forest

Preprocessing inside the pipeline:
- Numeric features → `StandardScaler`
- Categorical features (`Education`, `Marital_Status`) → `OneHotEncoder(handle_unknown="ignore")`

For each model it computes Accuracy, Precision, Recall, F1, ROC AUC, prints a classification report, and saves a confusion matrix PNG to `reports/`.

**Decision threshold tuning:** instead of using the default 0.5 cutoff, the script sweeps the precision-recall curve on the test set and picks the threshold that maximizes F1. This tuned threshold is saved separately and used by both the API and the dashboard at inference time.

**Outputs produced by training:**
- `models/model.pkl` — best-performing pipeline (full preprocessing + classifier)
- `models/decision_threshold.pkl` — tuned probability cutoff
- `reports/model_comparison.csv` — metrics for all models
- `reports/feature_importance.csv` — Random Forest feature importances
- `reports/<Model_Name>_confusion_matrix.png` — one per model
- `reports/Project_Report.pdf` — auto-generated summary report (via `report_generator.py`)

### How to run training
Training script uses **relative paths** (`../data`, `../models`, `../reports`), so it must be run **from inside `src/`**:
```bash
cd src
python train.py
```
This will take a few minutes due to the grid searches. Progress and metrics for each model print to the console.

---

## 6. FastAPI Inference Service (`api/main.py`)

A REST API that loads `models/model.pkl` and `models/decision_threshold.pkl` and serves predictions.

### Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Health/info message, shows loaded threshold |
| GET | `/health` | Health check |
| POST | `/predict` | Predict for a single customer |
| POST | `/predict_batch` | Predict for a list of customers |

### Request body for `/predict`
```json
{
  "Age": 45,
  "Income": 58000,
  "Education": "Graduation",
  "Marital_Status": "Single",
  "Children": 0,
  "Customer_Tenure": 1200,
  "Recency": 58,
  "NumDealsPurchases": 3,
  "NumWebVisitsMonth": 7,
  "Previous_Purchase": 12,
  "Total_Spending": 1500,
  "TotalAcceptedCmp": 0,
  "Complain": 0
}
```

### Example response
```json
{
  "prediction": "Likely Buyer",
  "confidence": 0.812,
  "probability_not_likely": 0.188,
  "probability_likely": 0.812,
  "threshold_used": 0.37,
  "features_used": ["Age", "Income", "Education", "..."]
}
```

`/predict_batch` accepts `{"customers": [ {...}, {...} ]}` and returns a `predictions` array plus a total count.

### How to run the API
The API resolves the model path relative to its own file location (`api/main.py` → `../models/model.pkl`), so launch it from the project root with uvicorn's module path:
```bash
uvicorn api.main:app --reload
```
Then open interactive Swagger docs at:
```
http://127.0.0.1:8000/docs
```

---

## 7. Streamlit Dashboard (`app.py`)

A full interactive dashboard combining EDA, live single-customer prediction, batch (CSV) prediction, and model performance reporting — all served locally and loading the model **directly** (not through the API).

### Features
- **KPI header** — total customers, average income, average spending, total likely buyers
- **Sidebar — Single Prediction**: sliders/inputs for all 13 features → click **Predict Customer** to get an instant prediction using the saved model + tuned threshold
- **Sidebar — Batch Prediction**: upload a CSV with the required columns to get predictions for many customers at once, with a downloadable results table
- **Tab 1 — Dataset Overview**: preview of cleaned data, shape, missing values, duplicate count, summary statistics
- **Tab 2 — Exploratory Data Analysis**: interactive Plotly charts (distributions, correlations, spending/response patterns)
- **Tab 3 — Model Performance**: model comparison table, confusion matrices, feature importance, and a button to regenerate the PDF report

### How to run the dashboard
Run from the project root (`korkai/`), since `app.py` loads `data/`, `models/`, and `src/` relative to its own location:
```bash
streamlit run app.py
```
This opens the dashboard in your browser, typically at:
```
http://localhost:8501
```

---

## 8. Setup & Installation (using `venv`)

### Step 1 — Create and activate a virtual environment

**Windows (PowerShell):**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

You should see `(venv)` appear at the start of your terminal prompt once it's active.

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — (Optional) Re-train the model
Model artifacts are already included in `models/`, so this step is only needed if you change the data, features, or want to regenerate everything from scratch.
```bash
cd src
python train.py
cd ..
```

### Step 4 — Run the API
```bash
uvicorn api.main:app --reload
```

### Step 5 — Run the Streamlit dashboard (in a separate terminal, with the same venv activated)
```bash
streamlit run app.py
```

### Deactivating the environment
When you're done:
```bash
deactivate
```

---

## 9. Tech Stack

- **Data processing:** pandas, numpy
- **Modeling:** scikit-learn (Logistic Regression, Decision Tree, Random Forest, Voting Classifier), XGBoost
- **Visualization:** matplotlib, seaborn, plotly
- **API:** FastAPI, uvicorn, pydantic
- **Dashboard:** Streamlit
- **Reporting:** reportlab (PDF generation), openpyxl
- **Model persistence:** joblib

---

## 10. Notes / Gotchas

- `train.py` and `preprocessing.py` use relative paths and are designed to be executed **from inside `src/`**.
- `app.py` and `api/main.py` resolve paths based on their own file location (`BASE_DIR`), so they should be launched from the project root as shown above — don't `cd` into `api/` or run `app.py` from another folder.
- The model pipeline already includes scaling and one-hot encoding internally, so raw feature values (not pre-scaled/encoded) should be passed to both the API and the dashboard.
- The **decision threshold is tuned**, not the default 0.5 — both the API and dashboard load `decision_threshold.pkl` and fall back to 0.5 only if that file is missing.