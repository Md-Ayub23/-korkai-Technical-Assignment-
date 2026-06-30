import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from src.report_generator import generate_report

# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="Customer Purchase Prediction System",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# CUSTOM CSS
# ==========================================================

st.markdown("""
<style>

.main{
    background-color:#f7f9fc;
}

.title{
    font-size:42px;
    font-weight:bold;
    color:#0E4C92;
}

.subtitle{
    color:gray;
    font-size:18px;
}

.metric-card{
    background:white;
    padding:18px;
    border-radius:12px;
    box-shadow:0px 3px 8px rgba(0,0,0,0.08);
}

.prediction-box{
    background:#ffffff;
    padding:25px;
    border-radius:15px;
    box-shadow:0px 4px 12px rgba(0,0,0,0.15);
}

.success-box{
    background:#d4edda;
    padding:15px;
    border-radius:10px;
}

.danger-box{
    background:#f8d7da;
    padding:15px;
    border-radius:10px;
}

hr{
    margin-top:30px;
    margin-bottom:30px;
}

</style>
""", unsafe_allow_html=True)

# ==========================================================
# LOAD MODEL & THRESHOLD
# ==========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

import sys
sys.path.append(os.path.join(BASE_DIR, "src"))

from preprocessing import preprocess

MODEL_PATH = os.path.join(BASE_DIR, "models", "model.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "decision_threshold.pkl")

model = joblib.load(MODEL_PATH)

# Load tuned decision threshold if available; fallback to 0.5
if os.path.exists(THRESHOLD_PATH):
    decision_threshold = joblib.load(THRESHOLD_PATH)
    threshold_loaded = True
else:
    decision_threshold = 0.5
    threshold_loaded = False

# Load already preprocessed dataset
df = preprocess(
    os.path.join(BASE_DIR, "data", "marketing_campaign.csv")
)


# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    '<p class="title">📊 Customer Purchase Prediction Dashboard</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">AI-Based Customer Data Analysis & Purchase Prediction System</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================================
# KPI CARDS
# ==========================================================

c1,c2,c3,c4 = st.columns(4)

with c1:
    st.metric("Customers", len(df))

with c2:
    st.metric("Average Income", f"${int(df['Income'].mean())}")

with c3:
    st.metric("Average Spending", f"${int(df['Total_Spending'].mean())}")

with c4:
    st.metric("Likely Buyers", int(df["Response"].sum()))

st.markdown("---")

# ==========================================================
# SIDEBAR INPUT
# ==========================================================

st.sidebar.header("Customer Information")

age = st.sidebar.slider("Age", 18, 90, 35)
income = st.sidebar.number_input("Income", 1000, 150000, 50000)
education = st.sidebar.selectbox("Education", sorted(df["Education"].unique()))
marital = st.sidebar.selectbox("Marital Status", sorted(df["Marital_Status"].unique()))
children = st.sidebar.slider("Children", 0, 5, 1)
tenure = st.sidebar.slider("Customer Tenure", 100, 6000, 3000)
recency = st.sidebar.slider("Recency", 0, 100, 30)
deals = st.sidebar.slider("Deals Purchased", 0, 15, 3)
visits = st.sidebar.slider("Web Visits / Month", 0, 20, 5)
purchase = st.sidebar.slider("Previous Purchases", 0, 40, 12)
spending = st.sidebar.number_input("Total Spending", 0, 5000, 800)

# NEW: TotalAcceptedCmp and Complain inputs
total_accepted_cmp = st.sidebar.slider("Total Accepted Campaigns", 0, 5, 0)
complain = st.sidebar.selectbox("Complain (0=No, 1=Yes)", [0, 1], index=0)

st.sidebar.markdown("---")
if threshold_loaded:
    st.sidebar.info(f"🔧 Tuned threshold: **{decision_threshold:.3f}**")
else:
    st.sidebar.warning("⚠️ Using default threshold 0.5")

predict = st.sidebar.button("Predict Customer", type="primary")

# ==========================================================
# CREATE INPUT DATAFRAME
# ==========================================================

input_df = pd.DataFrame({
    "Age": [age],
    "Income": [income],
    "Education": [education],
    "Marital_Status": [marital],
    "Children": [children],
    "Customer_Tenure": [tenure],
    "Recency": [recency],
    "NumDealsPurchases": [deals],
    "NumWebVisitsMonth": [visits],
    "Previous_Purchase": [purchase],
    "Total_Spending": [spending],
    "TotalAcceptedCmp": [total_accepted_cmp],   # NEW
    "Complain": [complain],                       # NEW
})

# ==========================================================
# PREDICTION
# ==========================================================

if predict:

    # Get probability of the positive class (purchase)
    probability = model.predict_proba(input_df)[0]
    prob_likely = float(probability[1])
    prob_not_likely = float(probability[0])

    # Apply tuned decision threshold instead of default 0.5
    prediction = 1 if prob_likely >= decision_threshold else 0
    confidence = prob_likely if prediction == 1 else prob_not_likely

    st.markdown("## Prediction Result")

    if prediction == 1:
        st.success(
            f"✅ Likely Buyer ({confidence:.2%} confidence)"
        )
    else:
        st.error(
            f"❌ Not Likely Buyer ({confidence:.2%} confidence)"
        )

    st.progress(float(confidence))

    # Show threshold info
    st.caption(f"Decision threshold used: **{decision_threshold:.3f}**")

    prob_df = pd.DataFrame({
        "Class": ["Not Likely Buyer", "Likely Buyer"],
        "Probability": [prob_not_likely, prob_likely]
    })

    fig = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        text=[f"{p:.3f}" for p in prob_df["Probability"]],
        title="Prediction Probability",
        color="Class",
        color_discrete_map={"Not Likely Buyer": "#e74c3c", "Likely Buyer": "#27ae60"}
    )
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, width="stretch")

    # Show input features summary
    with st.expander("View Input Features"):
        display_df = input_df.T.rename(columns={0: "Value"})
        display_df["Value"] = display_df["Value"].astype(str)
        st.dataframe(display_df, width="stretch")


# ==========================================================
# BATCH PREDICTION
# ==========================================================

st.sidebar.markdown("---")
st.sidebar.header("📦 Batch Prediction")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file with customer data",
    type=["csv"],
    help="CSV must contain columns: Age, Income, Education, Marital_Status, Children, Customer_Tenure, Recency, NumDealsPurchases, NumWebVisitsMonth, Previous_Purchase, Total_Spending, TotalAcceptedCmp, Complain"
)

if uploaded_file is not None:
    try:
        batch_df = pd.read_csv(uploaded_file)

        # Validate required columns
        required_cols = [
            "Age", "Income", "Education", "Marital_Status", "Children",
            "Customer_Tenure", "Recency", "NumDealsPurchases", "NumWebVisitsMonth",
            "Previous_Purchase", "Total_Spending", "TotalAcceptedCmp", "Complain"
        ]

        missing_cols = [col for col in required_cols if col not in batch_df.columns]

        if missing_cols:
            st.sidebar.error(f"❌ Missing columns: {', '.join(missing_cols)}")
        else:
            st.sidebar.success(f"✅ Loaded {len(batch_df)} customers")

            if st.sidebar.button("Run Batch Prediction", type="primary"):

                # Ensure correct column order
                batch_input = batch_df[required_cols]

                # Get probabilities
                batch_proba = model.predict_proba(batch_input)
                batch_prob_likely = batch_proba[:, 1]

                # Apply tuned threshold
                batch_predictions = (batch_prob_likely >= decision_threshold).astype(int)
                batch_confidences = np.where(batch_predictions == 1, batch_prob_likely, 1 - batch_prob_likely)

                # Create results dataframe
                results_df = batch_df.copy()
                results_df["Prediction"] = ["Likely Buyer" if p == 1 else "Not Likely Buyer" for p in batch_predictions]
                results_df["Confidence"] = np.round(batch_confidences, 3)
                results_df["Prob_Likely"] = np.round(batch_prob_likely, 3)
                results_df["Prob_Not_Likely"] = np.round(1 - batch_prob_likely, 3)
                results_df["Threshold_Used"] = round(decision_threshold, 3)

                st.markdown("---")
                st.markdown("## 📦 Batch Prediction Results")

                # Summary metrics
                total = len(results_df)
                likely_buyers = int(batch_predictions.sum())
                not_likely = total - likely_buyers
                avg_confidence = float(batch_confidences.mean())

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Customers", total)
                c2.metric("Likely Buyers", likely_buyers, f"{likely_buyers/total*100:.1f}%")
                c3.metric("Not Likely", not_likely, f"{not_likely/total*100:.1f}%")
                c4.metric("Avg Confidence", f"{avg_confidence:.2%}")

                st.markdown("---")

                # Results table
                st.subheader("Detailed Results")

                # Color-coded dataframe
                def color_prediction(val):
                    if val == "Likely Buyer":
                        return "background-color: #d4edda; color: #155724"
                    else:
                        return "background-color: #f8d7da; color: #721c24"

                display_cols = ["Age", "Income", "Education", "Marital_Status", "Children",
                               "Total_Spending", "TotalAcceptedCmp", "Prediction", "Confidence", "Prob_Likely"]

                styled_df = results_df[display_cols].style.map(color_prediction, subset=["Prediction"])
                st.dataframe(styled_df, width="stretch")

                # Distribution charts
                c1, c2 = st.columns(2)

                with c1:
                    pred_counts = results_df["Prediction"].value_counts().reset_index()
                    pred_counts.columns = ["Prediction", "Count"]
                    fig = px.pie(pred_counts, names="Prediction", values="Count",
                                 title="Prediction Distribution", color="Prediction",
                                 color_discrete_map={"Likely Buyer": "#27ae60", "Not Likely Buyer": "#e74c3c"})
                    st.plotly_chart(fig, width="stretch")

                with c2:
                    fig = px.histogram(results_df, x="Prob_Likely", nbins=20,
                                       title="Probability Distribution (Likely Buyer)",
                                       color_discrete_sequence=["#0E4C92"])
                    fig.add_vline(x=decision_threshold, line_dash="dash", line_color="red",
                                  annotation_text=f"Threshold: {decision_threshold:.3f}")
                    st.plotly_chart(fig, width="stretch")

                # Confidence by prediction
                fig = px.box(results_df, x="Prediction", y="Confidence", color="Prediction",
                             title="Confidence Distribution by Prediction",
                             color_discrete_map={"Likely Buyer": "#27ae60", "Not Likely Buyer": "#e74c3c"})
                st.plotly_chart(fig, width="stretch")

                # Download results
                csv_results = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Download Results CSV",
                    csv_results,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv",
                    type="primary"
                )

    except Exception as e:
        st.sidebar.error(f"❌ Error reading file: {str(e)}")

st.markdown("---")

# ==========================================================
# PDF REPORT GENERATION
# ==========================================================

st.divider()
st.subheader("📄 Project Report")

if st.button("Generate PDF Report", type="primary"):

    comparison_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")

    if os.path.exists(comparison_path):
        comparison = pd.read_csv(comparison_path)

        # Find best model by F1 Score (not accuracy, for consistency with train.py)
        best_idx = comparison["F1 Score"].idxmax()
        best_row = comparison.loc[best_idx]
        best_model_name = best_row["Model"]
        best_accuracy = best_row["Accuracy"]

        # Threshold info for report
        threshold_info = None
        if os.path.exists(THRESHOLD_PATH):
            threshold_info = {
                "threshold": decision_threshold,
                "accuracy": best_row["Accuracy"],  # placeholder; ideally from train.py output
                "precision": best_row["Precision"],
                "recall": best_row["Recall"],
                "f1": best_row["F1 Score"],
            }

        # Dataset stats for report
        dataset_stats = {
            "raw_records": "2,240",
            "clean_records": f"{df.shape[0]:,}",
            "response_rate": f"{df['Response'].mean():.1%}"
        }

        # Drop Best Params column if present (non-numeric, causes format errors)
        comparison_for_report = comparison.drop(columns=["Best Params"], errors="ignore")

        try:
            generate_report(
                comparison_for_report,
                best_model_name,
                best_accuracy,
                feature_importance_path=os.path.join(BASE_DIR, "reports", "feature_importance.csv"),
                threshold_info=threshold_info,
                dataset_stats=dataset_stats,
            )
            st.success("✅ Project_Report.pdf generated successfully! Scroll down to the 'Download Files' section to download it.")
        except PermissionError:
            st.error("❌ Permission denied: Close any open PDF viewer and try again.")
        except Exception as e:
            st.error(f"❌ Error generating report: {str(e)}")
    else:
        st.error("❌ model_comparison.csv not found. Run train.py first.")

# ==========================================================
# DASHBOARD TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs([
    "📊 Dataset Overview",
    "📈 Exploratory Data Analysis",
    "📉 Model Performance",
])

# ==========================================================
# TAB 1 – Dataset Overview
# ==========================================================

with tab1:

    st.subheader("Dataset Overview")

    c1, c2 = st.columns([2, 1])

    with c1:
        st.dataframe(df.head(15), width="stretch")

    with c2:
        st.write("### Dataset Information")
        st.metric("Rows", df.shape[0])
        st.metric("Columns", df.shape[1])
        st.metric("Missing Values", int(df.isnull().sum().sum()))
        st.metric("Duplicate Records", int(df.duplicated().sum()))

    st.markdown("---")
    st.subheader("Numerical Statistics")
    st.dataframe(df.describe(), width="stretch")

# ==========================================================
# TAB 2 – Exploratory Data Analysis
# ==========================================================

with tab2:

    st.subheader("Exploratory Data Analysis")

    # Age Distribution
    fig = px.histogram(df, x="Age", nbins=25, title="Age Distribution", color_discrete_sequence=["#0E4C92"])
    st.plotly_chart(fig, width="stretch")

    # Income Distribution
    fig = px.histogram(df, x="Income", nbins=30, title="Income Distribution", color_discrete_sequence=["#1A73C8"])
    st.plotly_chart(fig, width="stretch")

    # Spending Distribution
    fig = px.histogram(df, x="Total_Spending", nbins=30, title="Customer Spending Distribution", color_discrete_sequence=["#27AE60"])
    st.plotly_chart(fig, width="stretch")

    # Education
    edu = df["Education"].value_counts().reset_index()
    edu.columns = ["Education", "Customers"]
    fig = px.bar(edu, x="Education", y="Customers", text="Customers", title="Customer Distribution by Education")
    st.plotly_chart(fig, width="stretch")

    # Marital Status
    marital = df["Marital_Status"].value_counts().reset_index()
    marital.columns = ["Marital Status", "Customers"]
    fig = px.pie(marital, names="Marital Status", values="Customers", title="Marital Status Distribution")
    st.plotly_chart(fig, width="stretch")

    # Response
    response = df["Response"].value_counts().reset_index()
    response.columns = ["Buyer", "Count"]
    response["Buyer"] = response["Buyer"].replace({0: "Not Buyer", 1: "Buyer"})
    fig = px.bar(response, x="Buyer", y="Count", text="Count", title="Target Variable Distribution",
                 color="Buyer", color_discrete_map={"Not Buyer": "#e74c3c", "Buyer": "#27ae60"})
    st.plotly_chart(fig, width="stretch")

    # Income vs Spending
    fig = px.scatter(df, x="Income", y="Total_Spending", color=df["Response"].astype(str),
                     title="Income vs Spending", color_discrete_map={"0": "#e74c3c", "1": "#27ae60"})
    st.plotly_chart(fig, width="stretch")

    # Correlation Heatmap
    st.subheader("Feature Correlation")
    corr = df.select_dtypes(include="number").corr()
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="Blues", aspect="auto")
    st.plotly_chart(fig, width="stretch")

# ==========================================================
# TAB 3 – Model Performance
# ==========================================================

with tab3:

    st.subheader("Model Performance")

    comparison_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")

    if os.path.exists(comparison_path):

        comparison = pd.read_csv(comparison_path)
        st.dataframe(comparison, width="stretch")

        # All metrics comparison
        metrics = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"]
        for metric in metrics:
            if metric in comparison.columns:
                fig = px.bar(comparison, x="Model", y=metric, color="Model",
                             text=metric, title=f"Model {metric} Comparison")
                fig.update_traces(textposition="outside")
                st.plotly_chart(fig, width="stretch")

        # Best model highlight
        best_idx = comparison["F1 Score"].idxmax()
        best_row = comparison.loc[best_idx]
        st.info(f"🏆 Best Model (by F1 Score): **{best_row['Model']}** | F1: {best_row['F1 Score']:.4f} | Accuracy: {best_row['Accuracy']:.4f}")

    else:
        st.warning("model_comparison.csv not found.\nRun train.py first.")

st.markdown("---")

# ==========================================================
# PROJECT SUMMARY
# ==========================================================

st.header("🏆 Project Summary")

best_model = "Random Forest"

try:
    comparison = pd.read_csv(os.path.join(BASE_DIR, "reports", "model_comparison.csv"))

    # Use F1 Score for best model selection (consistent with train.py)
    best_row = comparison.loc[comparison["F1 Score"].idxmax()]

    best_model = best_row["Model"]
    accuracy = best_row["Accuracy"]
    precision = best_row["Precision"]
    recall = best_row["Recall"]
    f1 = best_row["F1 Score"]
    roc_auc = best_row.get("ROC AUC", "N/A")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Best Model", best_model)
    c2.metric("Accuracy", f"{accuracy:.2%}")
    c3.metric("Precision", f"{precision:.2%}")
    c4.metric("Recall", f"{recall:.2%}")
    c5.metric("F1 Score", f"{f1:.2%}")

    if roc_auc != "N/A":
        st.metric("ROC AUC", f"{roc_auc:.2%}")

except Exception as e:
    st.info("Run train.py to generate model comparison.")

st.markdown("---")

# ==========================================================
# CONFUSION MATRICES
# ==========================================================

st.header("📉 Confusion Matrices")

confusion_folder = os.path.join(BASE_DIR, "reports")

# Dynamically find all confusion matrix images
import glob
cm_images = sorted(glob.glob(os.path.join(confusion_folder, "*_confusion_matrix.png")))

if cm_images:
    cols = st.columns(min(len(cm_images), 3))
    for i, img_path in enumerate(cm_images):
        col = cols[i % len(cols)]
        img_name = os.path.basename(img_path).replace("_confusion_matrix.png", "").replace("_", " ")
        col.image(img_path, width="stretch", caption=img_name)
else:
    # Fallback to known model names
    images = [
        "Logistic_Regression_confusion_matrix.png",
        "Decision_Tree_confusion_matrix.png",
        "Random_Forest_confusion_matrix.png",
        "XGBoost_confusion_matrix.png",
        "Voting_Ensemble_(LR_+_RF)_confusion_matrix.png",
    ]
    cols = st.columns(min(len(images), 3))
    for col, img in zip(cols, images):
        img_path = os.path.join(confusion_folder, img)
        if os.path.exists(img_path):
            col.image(img_path, width="stretch", caption=img.replace("_", " ").replace(".png", ""))

st.markdown("---")

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

st.header("⭐ Feature Importance")

try:
    # Try loading from saved CSV first
    fi_path = os.path.join(BASE_DIR, "reports", "feature_importance.csv")
    if os.path.exists(fi_path):
        importance = pd.read_csv(fi_path).head(15)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                     title="Top 15 Important Features (Random Forest)", color_discrete_sequence=["#0E4C92"])
        st.plotly_chart(fig, width="stretch")
    else:
        # Fallback: extract from model pipeline
        classifier = model.named_steps["classifier"]
        feature_names = model.named_steps["preprocessor"].get_feature_names_out()

        if hasattr(classifier, "feature_importances_"):
            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance": classifier.feature_importances_
            }).sort_values("Importance", ascending=False).head(15)

            fig = px.bar(importance, x="Importance", y="Feature", orientation="h",
                         title="Top 15 Important Features", color_discrete_sequence=["#0E4C92"])
            st.plotly_chart(fig, width="stretch")
        else:
            st.info("Feature importance is available only for tree-based models.")

except Exception as e:
    st.info(f"Feature importance unavailable: {e}")

st.markdown("---")

# ==========================================================
# DOWNLOADS
# ==========================================================

st.header("📥 Download Files")

pdf_path = os.path.join(BASE_DIR, "reports", "Project_Report.pdf")
csv_path = os.path.join(BASE_DIR, "reports", "model_comparison.csv")
fi_csv_path = os.path.join(BASE_DIR, "reports", "feature_importance.csv")
model_path = os.path.join(BASE_DIR, "models", "model.pkl")
threshold_path = os.path.join(BASE_DIR, "models", "decision_threshold.pkl")

col1, col2, col3, col4 = st.columns(4)

if os.path.exists(model_path):
    with open(model_path, "rb") as f:
        col1.download_button("📦 Download Model", f, file_name="model.pkl")

if os.path.exists(csv_path):
    with open(csv_path, "rb") as f:
        col2.download_button("📊 Download CSV", f, file_name="model_comparison.csv")

if os.path.exists(fi_csv_path):
    with open(fi_csv_path, "rb") as f:
        col3.download_button("⭐ Feature Importance", f, file_name="feature_importance.csv")

if os.path.exists(pdf_path):
    with open(pdf_path, "rb") as f:
        col4.download_button("📄 Download Report", f, file_name="Project_Report.pdf")

if os.path.exists(threshold_path):
    with open(threshold_path, "rb") as f:
        st.download_button("🔧 Decision Threshold", f, file_name="decision_threshold.pkl")

st.markdown("---")

# ==========================================================
# ABOUT PROJECT
# ==========================================================

st.header("📘 About This Project")

st.markdown("""
### AI-Based Customer Purchase Prediction System

This application was developed as part of a Python AI/ML Technical Assessment for **Korkai Technologies**.

### System Components

- **Data Preprocessing** – Missing value imputation, outlier handling, feature encoding
- **Feature Engineering** – Age, Tenure, Total Spending, Previous Purchases, Total Accepted Campaigns
- **ML Training Pipeline** – Logistic Regression, Decision Tree, Random Forest, XGBoost, Voting Ensemble
- **Hyperparameter Tuning** – 5-fold Stratified Cross-Validation with GridSearchCV
- **Threshold Optimization** – Data-driven decision threshold via precision-recall curve
- **FastAPI REST API** – Real-time prediction endpoint with tuned threshold
- **Streamlit Dashboard** – Interactive EDA, prediction, and model performance visualization
- **PDF Report Generator** – Automated professional report generation

### Technologies Used

- **Python** | **Pandas** | **NumPy** | **Scikit-Learn** | **XGBoost**
- **FastAPI** | **Streamlit** | **Plotly** | **Joblib** | **ReportLab**

The system predicts whether a customer is likely to respond positively to a marketing campaign based on demographic information, purchasing behavior, and campaign history.
""")

st.markdown("---")

# ==========================================================
# FOOTER
# ==========================================================

st.markdown(
"""
<div style='text-align:center; padding:20px; color:gray;'>

Developed by <b>A. Mohammed Ayub</b><br>
Python AI/ML Developer Technical Assessment<br>
Korkai Technologies

</div>
""",
unsafe_allow_html=True
)