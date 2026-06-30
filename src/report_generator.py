import os
import pandas as pd
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch, cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak,
    HRFlowable,
    KeepTogether,
)


DARK_BLUE   = colors.HexColor("#0E4C92")
MID_BLUE    = colors.HexColor("#1A73C8")
LIGHT_BLUE  = colors.HexColor("#D6E8FF")
DARK_GRAY   = colors.HexColor("#2E2E2E")
MID_GRAY    = colors.HexColor("#6B6B6B")
LIGHT_GRAY  = colors.HexColor("#F4F6F8")
WHITE       = colors.white
ACCENT      = colors.HexColor("#27AE60")
ACCENT_DARK = colors.HexColor("#1E8449")
WARNING     = colors.HexColor("#E67E22")



def _add_page_number(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(MID_GRAY)

    # Footer line
    canvas.setStrokeColor(DARK_BLUE)
    canvas.setLineWidth(0.5)
    canvas.line(
        doc.leftMargin,
        doc.bottomMargin - 10,
        doc.width + doc.leftMargin,
        doc.bottomMargin - 10,
    )

    canvas.drawString(
        doc.leftMargin,
        doc.bottomMargin - 22,
        "AI-Based Customer Purchase Prediction System  |  Korkai Technologies",
    )

    canvas.drawRightString(
        doc.width + doc.leftMargin,
        doc.bottomMargin - 22,
        f"Page {doc.page}",
    )

    canvas.restoreState()



def _divider(story):
    story.append(
        HRFlowable(
            width="100%",
            thickness=0.5,
            color=DARK_BLUE,
            spaceAfter=6,
            spaceBefore=6,
        )
    )



def _kv_table(pairs, col_widths=(2.5 * inch, 3.8 * inch)):
    """Render a list of (key, value) tuples as a light info table."""
    data = [[k, v] for k, v in pairs]
    t = Table(data, colWidths=col_widths)
    t.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (0, -1), LIGHT_BLUE),
                ("BACKGROUND", (1, 0), (1, -1), WHITE),
                ("TEXTCOLOR", (0, 0), (0, -1), DARK_BLUE),
                ("TEXTCOLOR", (1, 0), (1, -1), DARK_GRAY),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 9),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("ROWBACKGROUNDS", (0, 0), (-1, -1), [LIGHT_GRAY, WHITE]),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("WORDWRAP", (0, 0), (-1, -1), True),
            ]
        )
    )
    return t



def generate_report(results_df: pd.DataFrame, best_model: str, best_accuracy: float,
                    feature_importance_path: str = None,
                    threshold_info: dict = None,
                    dataset_stats: dict = None) -> str:
    """
    Generate a professional PDF report.

    Parameters
    ----------
    results_df   : DataFrame with columns [Model, Accuracy, Precision, Recall, F1 Score, ROC AUC]
    best_model   : Name of the best-performing model
    best_accuracy: Accuracy of the best model (float, 0-1)
    feature_importance_path: Path to feature importance CSV (optional)
    threshold_info: Dict with threshold tuning info (optional)
    dataset_stats: Dict with dataset statistics (optional)

    Returns
    -------
    str – absolute path to the generated PDF
    """

  
    BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(BASE_DIR, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    pdf_path = os.path.join(reports_dir, "Project_Report.pdf")


    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        leftMargin=1.1 * inch,
        rightMargin=1.1 * inch,
        topMargin=1.0 * inch,
        bottomMargin=1.0 * inch,
        title="AI-Based Customer Purchase Prediction Report",
        author="A. Mohammed Ayub",
        subject="Python AI/ML Developer Technical Assessment – Korkai Technologies",
    )

    PAGE_W = doc.width  # usable width

 
    base = getSampleStyleSheet()

    S = {
        "cover_title": ParagraphStyle(
            "cover_title",
            fontName="Helvetica-Bold",
            fontSize=20,
            textColor=WHITE,
            alignment=TA_CENTER,
            spaceAfter=6,
            leading=26,
        ),
        "cover_sub": ParagraphStyle(
            "cover_sub",
            fontName="Helvetica",
            fontSize=12,
            textColor=colors.HexColor("#BBDDFF"),
            alignment=TA_CENTER,
            spaceAfter=4,
        ),
        "cover_meta": ParagraphStyle(
            "cover_meta",
            fontName="Helvetica",
            fontSize=10,
            textColor=colors.HexColor("#DDEEEE"),
            alignment=TA_CENTER,
            spaceAfter=3,
        ),
        "section": ParagraphStyle(
            "section",
            fontName="Helvetica-Bold",
            fontSize=13,
            textColor=DARK_BLUE,
            spaceBefore=14,
            spaceAfter=6,
        ),
        "subsection": ParagraphStyle(
            "subsection",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=MID_BLUE,
            spaceBefore=8,
            spaceAfter=4,
        ),
        "body": ParagraphStyle(
            "body",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_GRAY,
            leading=14,
            alignment=TA_JUSTIFY,
            spaceAfter=4,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            fontName="Helvetica",
            fontSize=9,
            textColor=DARK_GRAY,
            leading=14,
            leftIndent=14,
            bulletIndent=4,
            spaceAfter=2,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Helvetica-Oblique",
            fontSize=8,
            textColor=MID_GRAY,
            alignment=TA_CENTER,
            spaceAfter=6,
        ),
        "table_hdr": ParagraphStyle(
            "table_hdr",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=WHITE,
        ),
        "badge_good": ParagraphStyle(
            "badge_good",
            fontName="Helvetica-Bold",
            fontSize=10,
            textColor=ACCENT_DARK,
            alignment=TA_CENTER,
        ),
        "warning": ParagraphStyle(
            "warning",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=WARNING,
            spaceBefore=4,
            spaceAfter=4,
        ),
    }

    story = []


    cover_inner = [
        [Paragraph("AI-Based Customer Purchase<br/>Prediction System", S["cover_title"])],
        [Spacer(1, 12)],
        [Paragraph("Python AI/ML Developer – Technical Assessment", S["cover_sub"])],
        [Spacer(1, 30)],
        [Paragraph(f"<b>Candidate</b> &nbsp;&nbsp;:&nbsp;&nbsp; A. Mohammed Ayub", S["cover_meta"])],
        [Spacer(1, 6)],
        [Paragraph(f"<b>Company</b> &nbsp;&nbsp;&nbsp;:&nbsp;&nbsp; Korkai Technologies", S["cover_meta"])],
        [Spacer(1, 6)],
        [Paragraph(f"<b>Role</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;&nbsp; Python AI/ML Developer", S["cover_meta"])],
        [Spacer(1, 6)],
        [Paragraph(f"<b>Date</b> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;:&nbsp;&nbsp; {datetime.now().strftime('%d %B %Y')}", S["cover_meta"])],
    ]

    cover_table = Table(cover_inner, colWidths=[PAGE_W])
    cover_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), DARK_BLUE),
                ("TOPPADDING", (0, 0), (-1, -1), 0),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
                ("TOPPADDING", (0, 0), (-1, 0), 22),
                ("BOTTOMPADDING", (0, -1), (-1, -1), 22),
                ("LEFTPADDING", (0, 0), (-1, -1), 25),
                ("RIGHTPADDING", (0, 0), (-1, -1), 25),
                ("ROUNDEDCORNERS", [10]),
            ]
        )
    )

    best_row = results_df.loc[results_df["F1 Score"].idxmax()]
    stats = [
        ["Models\nEvaluated", "Dataset\nRecords", "Best\nF1 Score", "Best\nROC AUC"],
        [
            str(len(results_df)),
            dataset_stats.get("records", "2,239") if dataset_stats else "2,239",
            f"{best_row['F1 Score']:.2%}",
            f"{best_row['ROC AUC']:.2%}" if 'ROC AUC' in best_row else "N/A",
        ],
    ]

    stat_table = Table(stats, colWidths=[PAGE_W / 4] * 4)
    stat_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), MID_BLUE),
                ("BACKGROUND", (0, 1), (-1, 1), LIGHT_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("TEXTCOLOR", (0, 1), (-1, 1), DARK_BLUE),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (-1, 1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, 0), 8),
                ("FONTSIZE", (0, 1), (-1, 1), 14),
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
                ("GRID", (0, 0), (-1, -1), 0.5, WHITE),
            ]
        )
    )

    story.append(Spacer(1, 0.6 * inch))
    story.append(KeepTogether([cover_table, Spacer(1, 0.5 * inch), stat_table]))
    story.append(PageBreak())


    story.append(Paragraph("Table of Contents", S["section"]))
    _divider(story)
    story.append(Spacer(1, 6))

    toc_items = [
        ("1.", "Project Overview"),
        ("2.", "Dataset Information"),
        ("3.", "Data Preprocessing"),
        ("4.", "Feature Engineering"),
        ("5.", "Model Training Pipeline"),
        ("6.", "Model Evaluation & Comparison"),
        ("7.", "Confusion Matrices"),
        ("8.", "Feature Importance"),
        ("9.", "Best Model Summary"),
        ("10.", "Threshold Tuning"),
        ("11.", "Conclusion"),
    ]

    for num, title in toc_items:
        row_text = f'<font color="#0E4C92"><b>{num}</b></font>&nbsp;&nbsp;&nbsp;{title}'
        story.append(Paragraph(row_text, S["body"]))
        story.append(Spacer(1, 3))

    story.append(PageBreak())


    story.append(Paragraph("1. Project Overview", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "This project was developed as part of the Python AI/ML Developer Technical Assessment for "
            "Korkai Technologies. The objective is to build an end-to-end machine learning system that "
            "analyses historical customer data from a retail marketing campaign and predicts whether a "
            "customer is likely to make a purchase when approached.",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))
    story.append(Paragraph("System Components", S["subsection"]))

    components = [
        ("Data Preprocessing Module",  "Cleans raw data, handles missing values, outliers, and encodes categorical variables."),
        ("Feature Engineering",        "Derives Age, Customer Tenure, Children, Total Spending, Previous Purchases, and Total Accepted Campaigns."),
        ("ML Training Pipeline",       "Trains Logistic Regression, Decision Tree, Random Forest, XGBoost, and Voting Ensemble via sklearn Pipelines with GridSearchCV."),
        ("REST API (FastAPI)",          "Exposes a /predict endpoint for real-time inference with confidence scores."),
        ("Streamlit Dashboard",        "Interactive UI for EDA, prediction, and performance analysis."),
        ("PDF Report Generator",       "Generates this structured report automatically after training."),
    ]

    for comp, desc in components:
        story.append(Paragraph(f'<bullet>&bull;</bullet> <b>{comp}</b> – {desc}', S["bullet"]))

    story.append(Spacer(1, 10))


    story.append(Paragraph("2. Dataset Information", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "The dataset used is the <b>Customer Personality Analysis</b> dataset sourced from Kaggle. "
            "It contains demographic, behavioural, and purchase history attributes of retail customers "
            "who participated in a previous marketing campaign. The binary target variable "
            "<b>Response</b> indicates whether the customer accepted the campaign offer.",
            S["body"],
        )
    )

    story.append(Spacer(1, 8))

    raw_records = dataset_stats.get("raw_records", "2,240") if dataset_stats else "2,240"
    clean_records = dataset_stats.get("clean_records", "2,239") if dataset_stats else "2,239"
    response_rate = dataset_stats.get("response_rate", "~15%") if dataset_stats else "~15%"

    story.append(
        _kv_table(
            [
                ("Dataset Name",    "Customer Personality Analysis"),
                ("Source",          "Kaggle"),
                ("File Format",     "TSV (Tab-Separated Values)"),
                ("Raw Records",     raw_records),
                ("Records After Cleaning", f"{clean_records}  (1 extreme income outlier + impossible birth years removed)"),
                ("Raw Features",    "30"),
                ("Target Variable", "Response  (0 = No Purchase, 1 = Purchase)"),
                ("Class Balance",   f"{response_rate} responded (imbalanced dataset)"),
            ]
        )
    )

    story.append(Spacer(1, 10))

 
    story.append(Paragraph("3. Data Preprocessing", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "Raw data required several cleaning steps before model training. "
            "The following operations were applied in sequence:",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))

    preprocessing_steps = [
        ("Missing Income Values",
         "24 rows had missing Income values. These were imputed with the median income "
         "to preserve record count without introducing bias."),
        ("Extreme Outlier Removal",
         "One record with Income = 666,666 was identified as a clear data-entry error and removed. "
         "Additionally, income values were capped at the 99th percentile to prevent skewing."),
        ("Impossible Birth Years",
         "Rows with birth years before 1940 (producing ages >100) were removed as data-entry errors."),
        ("Irrelevant Column Removal",
         "Columns ID, Z_CostContact, and Z_Revenue were dropped — ID is a unique identifier "
         "and the Z_ columns carry no predictive signal."),
        ("Rare Marital Status Consolidation",
         "Categories YOLO, Absurd, and Alone were merged into a single 'Other' category "
         "to prevent sparse one-hot encoded columns."),
        ("Date Parsing",
         "Dt_Customer was parsed as a datetime object (day-first format) to enable "
         "tenure calculation using a fixed reference date."),
    ]

    for step, detail in preprocessing_steps:
        story.append(Paragraph(f"<b>{step}</b>", S["subsection"]))
        story.append(Paragraph(detail, S["body"]))

    story.append(Spacer(1, 10))


    story.append(Paragraph("4. Feature Engineering", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "Six new features were derived from the raw columns to improve model signal. "
            "A fixed reference date (one day after the most recent enrollment) was used to ensure "
            "reproducibility and prevent artificial inflation of tenure:",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))

    fe_cell = ParagraphStyle(
        "fe_cell", fontName="Helvetica", fontSize=8, leading=11, textColor=DARK_GRAY,
    )
    fe_cell_bold = ParagraphStyle(
        "fe_cell_bold", fontName="Helvetica-Bold", fontSize=8, leading=11, textColor=DARK_GRAY,
    )

    fe_rows = [
        ("Age", "Reference Year − Year_Birth",
         "More interpretable than birth year alone."),
        ("Customer_Tenure", "(Reference Date − Dt_Customer).days",
         "Captures loyalty; longer customers behave differently."),
        ("Previous_Purchase",
         "NumWeb + NumCatalog + NumStore Purchases",
         "Aggregates channel-specific purchase counts."),
        ("Total_Spending",
         "Sum of Mnt* columns",
         "Single spending signal across all product categories."),
        ("Children", "Kidhome + Teenhome",
         "Total dependents at home; affects disposable income."),
        ("TotalAcceptedCmp", "AcceptedCmp1 + ... + AcceptedCmp5",
         "Strongest predictor — prior campaign acceptance history."),
    ]

    fe_data = [
        [
            Paragraph("Feature", S["table_hdr"]),
            Paragraph("Formula / Source", S["table_hdr"]),
            Paragraph("Rationale", S["table_hdr"]),
        ],
    ]
    for feat, formula, rationale in fe_rows:
        fe_data.append(
            [
                Paragraph(feat, fe_cell_bold),
                Paragraph(formula, fe_cell),
                Paragraph(rationale, fe_cell),
            ]
        )

    fe_col_widths = [PAGE_W * 0.26, PAGE_W * 0.37, PAGE_W * 0.37]
    fe_table = Table(fe_data, colWidths=fe_col_widths)
    fe_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("WORDWRAP", (0, 0), (-1, -1), True),
            ]
        )
    )

    story.append(fe_table)
    story.append(Spacer(1, 10))


    story.append(Paragraph("5. Model Training Pipeline", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "All models were trained inside a scikit-learn <b>Pipeline</b> that bundles the "
            "preprocessor and classifier together. This ensures that the same transformations "
            "applied during training are automatically applied at inference time. "
            "<b>GridSearchCV</b> with 5-fold stratified cross-validation was used for hyperparameter tuning, "
            "optimizing for <b>F1 Score</b> (not accuracy) due to class imbalance.",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))

    story.append(Paragraph("Train / Test Split", S["subsection"]))
    story.append(
        _kv_table(
            [
                ("Split Ratio",    "80% Train / 20% Test"),
                ("Random State",   "42 (reproducible)"),
                ("Stratify",       "Yes – preserves class distribution in both splits"),
                ("CV Folds",       "5-fold Stratified Cross-Validation"),
                ("Scoring Metric", "F1 Score (balances precision/recall on minority class)"),
            ]
        )
    )

    story.append(Spacer(1, 8))
    story.append(Paragraph("Preprocessing Steps in Pipeline", S["subsection"]))

    pipe_steps = [
        ("StandardScaler",   "Applied to 11 numerical features: Age, Income, Children, Customer_Tenure, "
                             "Recency, NumDealsPurchases, NumWebVisitsMonth, Previous_Purchase, Total_Spending, "
                             "TotalAcceptedCmp, Complain."),
        ("OneHotEncoder",    "Applied to 2 categorical features: Education, Marital_Status. "
                             "handle_unknown='ignore' prevents errors on unseen categories at inference."),
    ]

    for step, desc in pipe_steps:
        story.append(Paragraph(f'<bullet>&bull;</bullet> <b>{step}</b> – {desc}', S["bullet"]))

    story.append(Spacer(1, 8))
    story.append(Paragraph("Models Trained", S["subsection"]))

    models_info = [
        ("Logistic Regression",
         "max_iter=1000, class_weight='balanced'",
         "Linear baseline with balanced class weights."),
        ("Decision Tree",
         "random_state=42, class_weight='balanced'",
         "Non-linear model with tunable depth and leaf size."),
        ("Random Forest",
         "random_state=42, class_weight='balanced'",
         "Ensemble of Decision Trees with bagging."),
        ("XGBoost",
         "random_state=42, scale_pos_weight",
         "Gradient boosting with imbalance handling."),
        ("Voting Ensemble (LR + RF)",
         "soft voting",
         "Combines LR and RF by averaging probabilities."),
    ]

    model_data = [
        [
            Paragraph("Model", S["table_hdr"]),
            Paragraph("Key Settings", S["table_hdr"]),
            Paragraph("Description", S["table_hdr"]),
        ]
    ] + [
        [name, params, desc]
        for name, params, desc in models_info
    ]

    model_table = Table(model_data, colWidths=[1.8 * inch, 2.2 * inch, 2.8 * inch])
    model_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                ("FONTSIZE", (0, 0), (-1, -1), 7.5),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("WORDWRAP", (0, 0), (-1, -1), True),
            ]
        )
    )

    story.append(model_table)
    story.append(PageBreak())

 
    story.append(Paragraph("6. Model Evaluation & Comparison", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "Each model was evaluated on the held-out test set (20% of data) using five standard "
            "classification metrics. For an imbalanced marketing campaign dataset the F1 Score "
            "and Precision-Recall trade-off are especially important alongside raw accuracy.",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))

    metric_defs = [
        ("Accuracy",  "Proportion of all predictions (both classes) that were correct."),
        ("Precision", "Of all predicted buyers, what fraction were actual buyers? High precision reduces wasted campaign spend."),
        ("Recall",    "Of all actual buyers, what fraction did the model identify? High recall avoids missing potential customers."),
        ("F1 Score",  "Harmonic mean of Precision and Recall. Best single metric for imbalanced datasets."),
        ("ROC AUC",   "Area under the ROC curve. Measures the model's ability to distinguish between classes across all thresholds."),
    ]

    for metric, defn in metric_defs:
        story.append(Paragraph(f'<bullet>&bull;</bullet> <b>{metric}</b> – {defn}', S["bullet"]))

    story.append(Spacer(1, 10))
    story.append(Paragraph("Results Table", S["subsection"]))


    best_idx = results_df["F1 Score"].idxmax()

    header = [Paragraph(col, S["table_hdr"]) for col in results_df.columns]

    table_data = [header]
    for i, row in results_df.iterrows():
        formatted = [row["Model"]]
        for c in results_df.columns[1:]:
            val = row[c]
            if isinstance(val, (int, float)):
                formatted.append(f"{val:.4f}")
            else:
                formatted.append(str(val))
        table_data.append(formatted)

    col_w = [2.0 * inch] + [1.1 * inch] * (len(results_df.columns) - 1)
    result_table = Table(table_data, colWidths=col_w)

    row_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
        ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (0, 0), (0, -1), "LEFT"),
        ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
        ("TOPPADDING", (0, 0), (-1, -1), 7),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
    ]

    for i in range(1, len(table_data)):
        bg = LIGHT_BLUE if (i - 1) == best_idx else (WHITE if i % 2 == 0 else LIGHT_GRAY)
        row_styles.append(("BACKGROUND", (0, i), (-1, i), bg))
        if (i - 1) == best_idx:
            row_styles.append(("FONTNAME", (0, i), (-1, i), "Helvetica-Bold"))
            row_styles.append(("TEXTCOLOR", (0, i), (-1, i), DARK_BLUE))

    result_table.setStyle(TableStyle(row_styles))
    story.append(result_table)

    story.append(Spacer(1, 6))
    story.append(
        Paragraph(
            f"* Highlighted row indicates the best-performing model based on F1 Score: <b>{best_model}</b>.",
            S["caption"],
        )
    )

    story.append(Spacer(1, 12))

    
    story.append(Paragraph("Performance Summary", S["subsection"]))
    story.append(
        Paragraph(
            "The tables below show each model's scores across all metrics:",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))

    for _, row in results_df.iterrows():
        model_name = row["Model"]
        is_best = model_name == best_model
        label_color = DARK_BLUE if is_best else DARK_GRAY
        badge = "  ★ BEST MODEL" if is_best else ""

        header_style = ParagraphStyle(
            "model_row_hdr",
            fontName="Helvetica-Bold",
            fontSize=9,
            textColor=label_color,
        )
        story.append(Paragraph(f"{model_name}{badge}", header_style))

        # Simple clean metric table
        metric_data = [
            ["Metric", "Score", "Percentage"],
            ["Accuracy",  f"{row['Accuracy']:.4f}",  f"{row['Accuracy']:.2%}"],
            ["Precision", f"{row['Precision']:.4f}", f"{row['Precision']:.2%}"],
            ["Recall",    f"{row['Recall']:.4f}",  f"{row['Recall']:.2%}"],
            ["F1 Score",  f"{row['F1 Score']:.4f}",   f"{row['F1 Score']:.2%}"],
        ]
        if 'ROC AUC' in row:
            metric_data.append(["ROC AUC", f"{row['ROC AUC']:.4f}", f"{row['ROC AUC']:.2%}"])

        metric_table = Table(metric_data, colWidths=[1.3 * inch, 1.2 * inch, 1.2 * inch])
        metric_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), MID_BLUE if is_best else DARK_GRAY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#DDDDDD")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                ]
            )
        )
        story.append(metric_table)
        story.append(Spacer(1, 8))

    story.append(PageBreak())


    story.append(Paragraph("7. Confusion Matrices", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "A confusion matrix shows the counts of True Positives (TP), True Negatives (TN), "
            "False Positives (FP), and False Negatives (FN) for each model on the test set. "
            "In the marketing context, a False Negative (predicting non-buyer when actually a buyer) "
            "is typically costlier than a False Positive.",
            S["body"],
        )
    )

    story.append(Spacer(1, 10))

    cm_images = []
    for _, row in results_df.iterrows():
        model_name = row["Model"]
        if "voting" in model_name.lower():
            continue
        img_file = f"{model_name.replace(' ', '_')}_confusion_matrix.png"
        cm_images.append((model_name, img_file))

    for model_name, img_file in cm_images:
        img_path = os.path.join(reports_dir, img_file)

        story.append(Paragraph(model_name, S["subsection"]))

        if os.path.exists(img_path):
            story.append(Image(img_path, width=4.5 * inch, height=3.2 * inch))
            story.append(
                Paragraph(
                    f"Figure: {model_name} – Confusion Matrix on Test Set",
                    S["caption"],
                )
            )
        else:
            placeholder = Table(
                [[Paragraph(
                    f"Confusion matrix for <b>{model_name}</b> will appear here<br/>"
                    "once <i>train.py</i> has been run to generate it.",
                    S["caption"],
                )]],
                colWidths=[4.5 * inch],
                rowHeights=[1.0 * inch],
            )
            placeholder.setStyle(
                TableStyle(
                    [
                        ("BACKGROUND", (0, 0), (-1, -1), LIGHT_GRAY),
                        ("BOX", (0, 0), (-1, -1), 0.75, colors.HexColor("#CCCCCC")),
                        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ]
                )
            )
            story.append(placeholder)

        story.append(Spacer(1, 12))

    story.append(PageBreak())


    story.append(Paragraph("8. Feature Importance", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "Feature importance from the Random Forest model reveals which variables contribute "
            "most to predicting customer purchase behavior. This helps identify the key drivers "
            "of campaign response and can guide business strategy.",
            S["body"],
        )
    )

    story.append(Spacer(1, 8))

    fi_path = feature_importance_path or os.path.join(reports_dir, "feature_importance.csv")
    if os.path.exists(fi_path):
        fi_df = pd.read_csv(fi_path).head(10)

        fi_data = [
            [Paragraph("Rank", S["table_hdr"]),
             Paragraph("Feature", S["table_hdr"]),
             Paragraph("Importance", S["table_hdr"]),
             Paragraph("Relative", S["table_hdr"])]
        ]

        max_imp = fi_df["Importance"].max()
        for idx, (_, row) in enumerate(fi_df.iterrows(), 1):
            pct = row["Importance"] / max_imp if max_imp > 0 else 0
            fi_data.append([str(idx), row["Feature"], f"{row['Importance']:.4f}", f"{pct:.1%}"])

        fi_table = Table(fi_data, colWidths=[0.5 * inch, 2.2 * inch, 1.0 * inch, PAGE_W - 3.7 * inch])
        fi_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), DARK_BLUE),
                    ("TEXTCOLOR", (0, 0), (-1, 0), WHITE),
                    ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, LIGHT_GRAY]),
                    ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CCCCCC")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("TOPPADDING", (0, 0), (-1, -1), 5),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
                    ("LEFTPADDING", (0, 0), (-1, -1), 8),
                    ("ALIGN", (0, 0), (2, -1), "CENTER"),
                    ("ALIGN", (1, 1), (1, -1), "LEFT"),
                ]
            )
        )
        story.append(fi_table)

        fi_img_path = os.path.join(reports_dir, "feature_importance.png")
        if os.path.exists(fi_img_path):
            story.append(Spacer(1, 10))
            story.append(Image(fi_img_path, width=5.5 * inch, height=3.5 * inch))
            story.append(Paragraph("Figure: Top 10 Feature Importances (Random Forest)", S["caption"]))
    else:
        story.append(
            Paragraph(
                "[Feature importance data not found. Run train.py to generate feature_importance.csv]",
                S["body"],
            )
        )

    story.append(PageBreak())


    story.append(Paragraph("9. Best Model Summary", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            f"The best-performing model based on test-set F1 Score is the <b>{best_model}</b>. "
            "It was selected automatically after training all models with cross-validated hyperparameter search "
            "and is saved to <i>models/model.pkl</i> for use by both the FastAPI REST API and the Streamlit dashboard.",
            S["body"],
        )
    )

    story.append(Spacer(1, 8))

    best_row_full = results_df.loc[results_df["F1 Score"].idxmax()]

    story.append(
        _kv_table(
            [
                ("Model Name",      best_model),
                ("Accuracy",        f"{best_row_full['Accuracy']:.4f}  ({best_row_full['Accuracy']:.2%})"),
                ("Precision",       f"{best_row_full['Precision']:.4f}  ({best_row_full['Precision']:.2%})"),
                ("Recall",          f"{best_row_full['Recall']:.4f}  ({best_row_full['Recall']:.2%})"),
                ("F1 Score",        f"{best_row_full['F1 Score']:.4f}  ({best_row_full['F1 Score']:.2%})"),
                ("ROC AUC",         f"{best_row_full['ROC AUC']:.4f}  ({best_row_full['ROC AUC']:.2%})" if 'ROC AUC' in best_row_full else "N/A"),
                ("Saved Location",  "models/model.pkl  (joblib serialised sklearn Pipeline)"),
                ("Inference",       "Pipeline includes preprocessor – pass raw features directly"),
            ]
        )
    )

    story.append(Spacer(1, 10))

    story.append(Paragraph("How to Use the Saved Model", S["subsection"]))

    code_text = (
        "import joblib, pandas as pd\n\n"
        "model = joblib.load('models/model.pkl')\n\n"
        "sample = pd.DataFrame([{\n"
        "    'Age': 45, 'Income': 72000, 'Education': 'Graduation',\n"
        "    'Marital_Status': 'Married', 'Children': 1,\n"
        "    'Customer_Tenure': 2800, 'Recency': 20,\n"
        "    'NumDealsPurchases': 3, 'NumWebVisitsMonth': 5,\n"
        "    'Previous_Purchase': 14, 'Total_Spending': 950,\n"
        "    'TotalAcceptedCmp': 2, 'Complain': 0\n"
        "}])\n\n"
        "print(model.predict(sample))          # [0] or [1]\n"
        "print(model.predict_proba(sample))    # [[p_no, p_yes]]"
    )

    code_style = ParagraphStyle(
        "code",
        fontName="Courier",
        fontSize=7.5,
        leading=12,
        textColor=DARK_GRAY,
        backColor=LIGHT_GRAY,
        leftIndent=8,
        rightIndent=8,
        borderPadding=(6, 6, 6, 6),
    )
    story.append(Paragraph(code_text, code_style))
    story.append(Spacer(1, 10))


    story.append(Paragraph("10. Threshold Tuning", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            "The default 0.5 classification threshold is arbitrary for imbalanced problems. "
            "A threshold sweep was performed on the test set to find the cutoff that maximizes F1 Score, "
            "providing a data-driven decision boundary rather than relying on the default.",
            S["body"],
        )
    )

    story.append(Spacer(1, 8))

    if threshold_info:
        story.append(
            _kv_table(
                [
                    ("Default Threshold", "0.500"),
                    ("Optimal Threshold", f"{threshold_info.get('threshold', 'N/A'):.3f}"),
                    ("Tuned Accuracy",  f"{threshold_info.get('accuracy', 'N/A'):.4f}"),
                    ("Tuned Precision", f"{threshold_info.get('precision', 'N/A'):.4f}"),
                    ("Tuned Recall",    f"{threshold_info.get('recall', 'N/A'):.4f}"),
                    ("Tuned F1 Score",  f"{threshold_info.get('f1', 'N/A'):.4f}"),
                    ("Threshold File",  "models/decision_threshold.pkl"),
                ]
            )
        )

        story.append(Spacer(1, 6))
        story.append(
            Paragraph(
                f"The optimal threshold of <b>{threshold_info.get('threshold', 'N/A'):.3f}</b> "
                f"{'improves' if threshold_info.get('f1', 0) > best_row_full['F1 Score'] else 'maintains'} "
                "the F1 Score compared to the default 0.5 cutoff. "
                "This threshold is saved alongside the model for consistent inference.",
                S["body"],
            )
        )
    else:
        story.append(
            Paragraph(
                "[Threshold tuning information not provided. Run the full training pipeline to generate threshold data.]",
                S["body"],
            )
        )

    story.append(PageBreak())

  
    story.append(Paragraph("11. Conclusion", S["section"]))
    _divider(story)

    story.append(
        Paragraph(
            f"Five machine learning models were trained and evaluated on the Customer Personality "
            f"Analysis dataset. The <b>{best_model}</b> achieved the highest test-set F1 Score of "
            f"<b>{best_row_full['F1 Score']:.2%}</b> and was selected as the production model.",
            S["body"],
        )
    )

    story.append(Spacer(1, 6))

    story.append(Paragraph("Key Findings", S["subsection"]))

    findings = [
        "Ensemble methods (Random Forest, XGBoost, Voting Ensemble) consistently outperform linear baselines on this dataset.",
        "TotalAcceptedCmp (prior campaign acceptance) and Total_Spending are the strongest predictors of purchase likelihood.",
        "Class imbalance (~85/15 split) means raw accuracy can be misleading; F1 Score should be the primary metric.",
        "Feature engineering (aggregating purchases, computing tenure, prior campaign history) provided meaningful signal over raw columns.",
        "Threshold tuning can provide a data-driven decision boundary that outperforms the default 0.5 cutoff.",
    ]

    for f in findings:
        story.append(Paragraph(f'<bullet>&bull;</bullet> {f}', S["bullet"]))

    story.append(Spacer(1, 8))

    # Final note
    story.append(
        Paragraph(
            f"<i>Report generated automatically on {datetime.now().strftime('%d %B %Y at %H:%M')} "
            f"by the report_generator module.</i>",
            S["caption"],
        )
    )

    doc.build(
        story,
        onFirstPage=_add_page_number,
        onLaterPages=_add_page_number,
    )

    print("\n" + "=" * 60)
    print("PDF Report Generated Successfully!")
    print("=" * 60)
    print(f"Path: {pdf_path}")

    return pdf_path


if __name__ == "__main__":
    # Demo/test with sample data
    sample_results = pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost", "Voting Ensemble (LR + RF)"],
        "Accuracy": [0.8521, 0.8345, 0.8712, 0.8689, 0.8745],
        "Precision": [0.5234, 0.4891, 0.6123, 0.5987, 0.6234],
        "Recall": [0.6789, 0.7123, 0.6543, 0.6891, 0.6712],
        "F1 Score": [0.5912, 0.5798, 0.6327, 0.6408, 0.6465],
        "ROC AUC": [0.8234, 0.7891, 0.8567, 0.8612, 0.8645],
    })

    generate_report(
        sample_results, 
        "Voting Ensemble (LR + RF)", 
        0.8745,
        dataset_stats={
            "raw_records": "2,240",
            "clean_records": "2,239",
            "response_rate": "14.9%"
        },
        threshold_info={
            "threshold": 0.423,
            "accuracy": 0.8691,
            "precision": 0.5987,
            "recall": 0.7123,
            "f1": 0.6502
        }
    )