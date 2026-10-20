#!/usr/bin/env python3
"""
Generate 10-page academic PDF report for MLOps Assignment 1.
Uses ReportLab for layout.
"""

from pathlib import Path
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm, mm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, HRFlowable, KeepTogether, Preformatted
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfgen import canvas
from reportlab.lib.colors import HexColor

BASE = Path("/Users/ajitjena/Desktop/Master Program/Semester 3/MLOps/Assignment/Assignment 1")
FIGS = BASE / "reports" / "figures"
SHOTS = BASE / "reports" / "screenshots"
OUT_PDF = BASE / "reports" / "FINAL_REPORT.pdf"

# ── Colours ────────────────────────────────────────────────────────────────
C_DARK   = HexColor("#1A237E")   # deep navy
C_MID    = HexColor("#283593")   # mid blue
C_ACCENT = HexColor("#1565C0")   # bright blue
C_LIGHT  = HexColor("#E8EAF6")   # lavender tint
C_ORANGE = HexColor("#E65100")   # deep orange
C_GRAY   = HexColor("#37474F")   # blue-grey
C_LINE   = HexColor("#90A4AE")   # soft divider

W, H = A4

# ── Styles ─────────────────────────────────────────────────────────────────
base_styles = getSampleStyleSheet()

def make_style(name, parent="Normal", **kw):
    return ParagraphStyle(name, parent=base_styles[parent], **kw)

styles = {
    "title":    make_style("S_Title",   "Title",   fontSize=22, textColor=C_DARK,
                           spaceAfter=4, fontName="Helvetica-Bold", alignment=TA_CENTER),
    "subtitle": make_style("S_Sub",     "Normal",  fontSize=11, textColor=C_MID,
                           spaceAfter=2, fontName="Helvetica", alignment=TA_CENTER),
    "meta":     make_style("S_Meta",    "Normal",  fontSize=9,  textColor=C_GRAY,
                           spaceAfter=2, fontName="Helvetica", alignment=TA_CENTER),
    "h1":       make_style("S_H1",      "Heading1", fontSize=13, textColor=C_DARK,
                           spaceBefore=10, spaceAfter=4, fontName="Helvetica-Bold"),
    "h2":       make_style("S_H2",      "Heading2", fontSize=11, textColor=C_MID,
                           spaceBefore=6, spaceAfter=3, fontName="Helvetica-Bold"),
    "body":     make_style("S_Body",    "Normal",  fontSize=9,  textColor=C_GRAY,
                           leading=13, spaceAfter=4, alignment=TA_JUSTIFY),
    "bullet":   make_style("S_Bullet",  "Normal",  fontSize=8.5, textColor=C_GRAY,
                           leading=12, leftIndent=12, spaceAfter=2),
    "caption":  make_style("S_Cap",     "Normal",  fontSize=7.5, textColor=C_GRAY,
                           alignment=TA_CENTER, spaceAfter=4, fontName="Helvetica-Oblique"),
    "code":     make_style("S_Code",    "Code",    fontSize=7.5, textColor=HexColor("#263238"),
                           backColor=HexColor("#F5F5F5"), fontName="Courier", leading=11,
                           leftIndent=6, rightIndent=6, spaceAfter=4),
    "note":     make_style("S_Note",    "Normal",  fontSize=8,  textColor=HexColor("#4CAF50"),
                           fontName="Helvetica-Bold"),
    "placeholder": make_style("S_PH",  "Normal",  fontSize=9,  textColor=HexColor("#9E9E9E"),
                           alignment=TA_CENTER, fontName="Helvetica-Oblique",
                           borderColor=HexColor("#BDBDBD"), borderWidth=1, borderPadding=8),
}

# ── Helper: scaled image ────────────────────────────────────────────────────
def fig(path, max_w, max_h=None, caption=None):
    """Return image + optional caption flowables."""
    items = []
    if Path(path).exists():
        img = Image(str(path))
        iw, ih = img.imageWidth, img.imageHeight
        scale = min(max_w / iw, (max_h or 9999) / ih, 1.0)
        img.drawWidth  = iw * scale
        img.drawHeight = ih * scale
        items.append(img)
    else:
        items.append(Paragraph(f"[Image placeholder: {Path(path).name}]", styles["placeholder"]))
    if caption:
        items.append(Spacer(1, 1*mm))
        items.append(Paragraph(caption, styles["caption"]))
    return items

def ph(label, w=None, h=3.5*cm):
    """Placeholder box for screenshots the student will insert."""
    box = Table(
        [[Paragraph(label, styles["placeholder"])]],
        colWidths=[w or (W - 3.2*cm)],
        rowHeights=[h]
    )
    box.setStyle(TableStyle([
        ("BOX",        (0,0), (-1,-1), 0.8, C_LINE),
        ("BACKGROUND", (0,0), (-1,-1), HexColor("#FAFAFA")),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
    ]))
    return box

def hr(color=C_LINE, thickness=0.5):
    return HRFlowable(width="100%", thickness=thickness, color=color, spaceAfter=4, spaceBefore=2)

def two_col(left_items, right_items, split=0.5):
    """Two-column layout using a Table."""
    col_w = W - 3.2*cm
    lw = col_w * split - 3*mm
    rw = col_w * (1-split) - 3*mm
    return Table([[left_items, right_items]], colWidths=[lw, rw],
                 style=TableStyle([("VALIGN", (0,0), (-1,-1), "TOP"),
                                   ("LEFTPADDING",  (0,0), (-1,-1), 0),
                                   ("RIGHTPADDING", (0,0), (-1,-1), 0)]))

def code_block(text):
    """Render preformatted code with background box, preserving newlines and indentation."""
    from reportlab.lib.styles import ParagraphStyle as PS
    pre_style = PS(
        "CodePre",
        fontName="Courier",
        fontSize=7.5,
        leading=11,
        textColor=HexColor("#263238"),
        backColor=HexColor("#F5F5F5"),
        leftIndent=6,
        rightIndent=6,
        spaceBefore=2,
        spaceAfter=4,
    )
    box = Table(
        [[Preformatted(text, pre_style)]],
        colWidths=[W - 3.2*cm]
    )
    box.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), HexColor("#F5F5F5")),
        ("BOX",          (0,0), (-1,-1), 0.5, HexColor("#BDBDBD")),
        ("LEFTPADDING",  (0,0), (-1,-1), 6),
        ("RIGHTPADDING", (0,0), (-1,-1), 6),
        ("TOPPADDING",   (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
    ]))
    return box

# ── Page-number footer ──────────────────────────────────────────────────────
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        page_num = self._saved_page_states.index(dict(self.__dict__)) + 1 \
            if dict(self.__dict__) in self._saved_page_states else "?"
        # Footer bar
        self.setFillColor(C_DARK)
        self.rect(0, 0, W, 14*mm, fill=1, stroke=0)
        self.setFillColor(colors.white)
        self.setFont("Helvetica", 7.5)
        self.drawCentredString(W/2, 5*mm, f"HeartGuard — MLOps Assignment 1  |  Jena Ajit Anirudha Ranjita (2025AE05839)  |  BITS Pilani  |  Page {self._pageNumber} of {page_count}")
        # Header rule
        self.setStrokeColor(C_ACCENT)
        self.setLineWidth(2)
        self.line(1.6*cm, H - 1.4*cm, W - 1.6*cm, H - 1.4*cm)


# ── Build story ─────────────────────────────────────────────────────────────
story = []

# ══════════════════════════════════════════════════════════════════
# PAGE 1: Cover / Title
# ══════════════════════════════════════════════════════════════════
story.append(Spacer(1, 1.5*cm))
story.append(Paragraph("HeartGuard", styles["title"]))
story.append(Paragraph("An Observable End-to-End MLOps Pipeline for Cardiovascular Risk Prediction", styles["subtitle"]))
story.append(Spacer(1, 5*mm))
story.append(hr(C_ACCENT, 1.5))

meta_data = [
    ["Student", "Jena Ajit Anirudha Ranjita"],
    ["Student ID", "2025AE05839"],
    ["Course", "AIMLCZG523 — Machine Learning Operations (MLOps)"],
    ["Institution", "Birla Institute of Technology & Science (BITS), Pilani"],
    ["Assignment", "Experimental Learning Assignment 01  (Total Marks: 50)"],
    ["Dataset", "Heart Disease UCI — Cleveland Clinic Foundation Variant"],
    ["Model Serving", "FastAPI · Docker · Kubernetes (Minikube) · Helm"],
    ["Best Model", "Logistic Regression  |  ROC-AUC: 0.9632  |  Recall: 89.29%"],
]
meta_table = Table(meta_data, colWidths=[4.2*cm, 12*cm])
meta_table.setStyle(TableStyle([
    ("FONTNAME",    (0,0), (0,-1), "Helvetica-Bold"),
    ("FONTNAME",    (1,0), (1,-1), "Helvetica"),
    ("FONTSIZE",    (0,0), (-1,-1), 9),
    ("TEXTCOLOR",   (0,0), (0,-1), C_DARK),
    ("TEXTCOLOR",   (1,0), (1,-1), C_GRAY),
    ("ROWBACKGROUNDS", (0,0), (-1,-1), [C_LIGHT, colors.white]),
    ("TOPPADDING",  (0,0), (-1,-1), 4),
    ("BOTTOMPADDING",(0,0),(-1,-1), 4),
    ("LEFTPADDING", (0,0), (-1,-1), 6),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
]))
story.append(Spacer(1, 5*mm))
story.append(meta_table)
story.append(Spacer(1, 6*mm))

# Architecture diagram
story += fig(FIGS / "architecture_diagram.png", W - 3.2*cm, 9*cm,
             "Figure 0 — End-to-End MLOps System Architecture")

story.append(Spacer(1, 5*mm))
story.append(Paragraph(
    "This report documents the complete MLOps pipeline for predicting coronary artery disease risk from "
    "patient biomarkers. The system spans automated data ingestion, zero-leakage feature engineering, "
    "stratified model tuning with MLflow tracking, FastAPI model serving, Docker containerization, "
    "Kubernetes orchestration, and a real-time drift detection engine. <b>Recall (sensitivity) is "
    "prioritised</b> throughout—a false negative in cardiac screening can be fatal.",
    styles["body"]
))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 2: Task 1 — Data Acquisition & EDA
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("1 · Data Acquisition & Exploratory Data Analysis", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>1.1 Dataset Overview</b>", styles["h2"]))
story.append(Paragraph(
    "The <b>Cleveland Clinic Foundation Heart Disease Dataset</b> (UCI ML Repository, CC BY 4.0) "
    "contains <b>303 patient records</b> across 14 standardised clinical attributes collected by "
    "Robert Detrano, M.D., Ph.D. et al. The binary target encodes angiographic coronary artery disease "
    "(0 = absence, 1 = ≥50% diameter narrowing). Raw multi-class labels (0–4) are collapsed to binary "
    "following clinical screening convention.",
    styles["body"]
))

# Feature table
feat_rows = [
    ["Feature", "Type", "Clinical Meaning"],
    ["age", "Continuous", "Patient age in years (29–77)"],
    ["sex", "Binary", "Biological sex: 1=Male, 0=Female"],
    ["cp", "Ordinal (1–4)", "Chest pain: 1=Typical angina, 2=Atypical, 3=Non-anginal, 4=Asymptomatic"],
    ["trestbps", "Continuous", "Resting systolic blood pressure (mm Hg)"],
    ["chol", "Continuous", "Serum cholesterol (mg/dl)"],
    ["fbs", "Binary", "Fasting blood sugar >120 mg/dl (1=Yes)"],
    ["restecg", "Ordinal (0–2)", "Resting ECG result"],
    ["thalach", "Continuous", "Maximum heart rate achieved (bpm)"],
    ["exang", "Binary", "Exercise-induced angina (1=Yes)"],
    ["oldpeak", "Continuous", "ST depression (exercise vs rest, mm)"],
    ["slope", "Ordinal (1–3)", "Slope of peak exercise ST segment"],
    ["ca", "Ordinal (0–3)", "Major vessels coloured by fluoroscopy"],
    ["thal", "Categorical", "Thalassemia: 3=Normal, 6=Fixed, 7=Reversible defect"],
    ["target", "Binary", "Label: 0=No disease, 1=Disease present"],
]
ft = Table(feat_rows, colWidths=[2.4*cm, 3.2*cm, 10.6*cm])
ft.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0), C_DARK),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 7.5),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_LIGHT, colors.white]),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
    ("TOPPADDING",  (0,0), (-1,-1), 2),
    ("BOTTOMPADDING",(0,0),(-1,-1),2),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
]))
story.append(ft)
story.append(Spacer(1, 3*mm))

story.append(Paragraph("<b>1.2 Cleaning Pipeline</b> (<i>src/data/clean_data.py</i>)", styles["h2"]))
bullets = [
    "Schema validation: confirms all 14 columns with correct dtypes.",
    "Duplicate removal: 0 duplicate rows found.",
    "Binary target encoding: {1,2,3,4} → 1  (164 Absent · 139 Present).",
    "Physiological bounds check: age 18–120, trestbps 70–260, chol 80–650, thalach 50–250, oldpeak 0–10.",
    "Leakage-safe imputation: ca (4 nulls) and thal (2 nulls) kept as NaN — imputed inside CV folds only.",
]
for b in bullets:
    story.append(Paragraph(f"• {b}", styles["bullet"]))
story.append(Spacer(1, 2*mm))

# EDA figures — two per row
story.append(Paragraph("<b>1.3 EDA Visualisations</b>", styles["h2"]))

row1 = fig(FIGS / "01_target_distribution.png", (W-3.6*cm)*0.55, 4.8*cm,
           "Fig 1 — Target Distribution (Lollipop + Donut)")
row1b = fig(FIGS / "06_chest_pain_vs_disease.png", (W-3.6*cm)*0.43, 4.8*cm,
            "Fig 6 — Chest Pain Type vs. Disease")
story.append(two_col(row1, row1b, 0.56))
story.append(Spacer(1, 2*mm))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 3: EDA continued
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("1 · Exploratory Data Analysis (continued)", styles["h1"]))
story.append(hr())

story += fig(FIGS / "02_numerical_histograms.png", W - 3.2*cm, 4.5*cm,
             "Fig 2 — Numerical Feature Distributions split by Diagnosis Class (Histogram + KDE)")
story.append(Spacer(1, 2*mm))

story += fig(FIGS / "03_categorical_distributions.png", W - 3.2*cm, 4.5*cm,
             "Fig 3 — Categorical Feature Distributions — Grouped Bar (No Disease vs. Disease)")
story.append(Spacer(1, 2*mm))

row2a = fig(FIGS / "04_correlation_heatmap.png", (W-3.6*cm)*0.52, 5.5*cm,
            "Fig 4 — Spearman Rank Correlation Heatmap")
row2b = fig(FIGS / "08_fluoroscopy_vessels_vs_target.png", (W-3.6*cm)*0.46, 5.5*cm,
            "Fig 8 — Fluoroscopy Vessels & Thalassemia vs. Disease %")
story.append(two_col(row2a, row2b, 0.53))
story.append(Spacer(1, 2*mm))

row2c = fig(FIGS / "05_feature_vs_target_violin.png", (W-3.6*cm)*0.52, 4.5*cm,
            "Fig 5 — Box + Strip Plot: Numerical Features vs. Diagnosis Class")
row2d = fig(FIGS / "07_thalach_vs_age_scatter.png", (W-3.6*cm)*0.46, 4.5*cm,
            "Fig 7 — Max Heart Rate vs. Age (Scatter + 2D KDE)")
story.append(two_col(row2c, row2d, 0.53))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 4: Task 2 — Feature Engineering & Model Development
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("2 · Feature Engineering & Model Development", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>2.1 Zero-Leakage ColumnTransformer Pipeline</b> (<i>src/features/build_features.py</i>)", styles["h2"]))
story.append(Paragraph(
    "All transformations sit <i>inside</i> the scikit-learn Pipeline so they are fit exclusively on "
    "training folds and never see test data. Raw 13 features → <b>28 encoded features</b>.",
    styles["body"]
))
pipeline_code = """\
Numerical branch  [age, trestbps, chol, thalach, oldpeak]
  +-- SimpleImputer(strategy='median')  ->  StandardScaler()

Categorical branch  [sex, cp, fbs, restecg, exang, slope, ca, thal]
  +-- SimpleImputer(strategy='most_frequent')  ->  OneHotEncoder(handle_unknown='ignore')"""
story.append(code_block(pipeline_code))

story.append(Paragraph("<b>2.2 Model Selection & Hyperparameter Tuning</b>", styles["h2"]))
story.append(Paragraph(
    "Three candidate architectures were evaluated under <b>Stratified 5-Fold GridSearchCV</b> "
    "(seed=42, scoring=roc_auc). The <i>clinical selection score</i> weights Recall 60% + ROC-AUC 40% "
    "to minimise fatal False Negatives.",
    styles["body"]
))

perf_rows = [
    ["Model", "5-Fold CV AUC", "Test Accuracy", "Test Recall", "Test Precision", "Test F1", "Test AUC", "FN / 28", "Selected"],
    ["Logistic Regression", "0.9029", "86.89%", "89.29%", "83.33%", "0.8621", "0.9632", "3", "✓ YES"],
    ["Random Forest",       "0.9027", "88.52%", "89.29%", "86.21%", "0.8772", "0.9470", "3", "Baseline"],
    ["Gradient Boosting",   "0.8847", "86.89%", "85.71%", "85.71%", "0.8571", "0.9280", "4", "Bench."],
]
pt = Table(perf_rows, colWidths=[3.6*cm,2*cm,2*cm,2*cm,2.2*cm,1.6*cm,1.6*cm,1.2*cm,1.6*cm])
pt.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0), C_DARK),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 7.5),
    ("BACKGROUND",  (0,1), (-1,1), HexColor("#E3F2FD")),
    ("FONTNAME",    (-1,1), (-1,1), "Helvetica-Bold"),
    ("ROWBACKGROUNDS",(0,2),(-1,-1),[colors.white, C_LIGHT]),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
    ("TOPPADDING",  (0,0), (-1,-1), 3),
    ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING", (0,0), (-1,-1), 3),
    ("ALIGN",       (1,0), (-1,-1), "CENTER"),
]))
story.append(pt)
story.append(Spacer(1, 3*mm))

story.append(Paragraph("<b>Why Logistic Regression Won:</b>", styles["h2"]))
reasons = [
    "<b>Superior ROC-AUC (0.9632 vs 0.9470)</b>: Best discriminative capacity across all thresholds.",
    "<b>Fewest False Negatives</b>: Misses only 3/28 cardiac patients (matches RF but with better AUC).",
    "<b>Clinical Interpretability</b>: Log-odds coefficients translate directly to odds ratios.",
    "<b>Latency</b>: Sub-millisecond inference (<1 ms per sample) — ideal for real-time API serving.",
]
for r in reasons:
    story.append(Paragraph(f"• {r}", styles["bullet"]))
story.append(Spacer(1, 3*mm))

# ROC + confusion matrix side by side
row3a = fig(FIGS / "roc_curve.png", (W-3.6*cm)*0.5, 5.0*cm,
            "Fig 9 — Combined ROC Curves (All 3 Models)")
row3b = fig(FIGS / "confusion_matrix.png", (W-3.6*cm)*0.48, 5.0*cm,
            "Fig 10 — Confusion Matrices (LR vs RF, held-out test N=61)")
story.append(two_col(row3a, row3b, 0.51))
story.append(Spacer(1, 2*mm))

story += fig(FIGS / "feature_importance_logisticregression.png", W - 3.2*cm, 4.5*cm,
             "Fig 11 — Top 15 Feature Coefficients (Logistic Regression, Log-Odds Scale)")

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 5: Task 3 — MLflow Experiment Tracking
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("3 · MLflow Experiment Tracking", styles["h1"]))
story.append(hr())

story.append(Paragraph(
    "Every training run is tracked in an <b>MLflow experiment</b> (<i>heart-disease-classification</i>) "
    "backed by a local SQLite store (<code>mlruns/mlflow.db</code>). The following are logged per run:",
    styles["body"]
))
mlflow_items = [
    "<b>Parameters</b>: model_type, cv_folds, random_state, train/test sizes, scoring metric, all GridSearchCV hyperparameters.",
    "<b>Metrics</b>: cv_best_roc_auc, test_accuracy, test_recall, test_precision, test_f1, test_roc_auc, test_false_negatives.",
    "<b>Artifacts</b>: confusion matrix PNG, feature importance PNG, classification report JSON, full serialised Pipeline.",
    "<b>Model</b>: MLflow pyfunc bundle with input signature and example, stored as <code>models/mlflow_model/</code>.",
    "<b>Tags</b>: source notebook, source type, run name.",
]
for item in mlflow_items:
    story.append(Paragraph(f"• {item}", styles["bullet"]))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("Screenshot — MLflow Experiments UI", styles["h2"]))
if (SHOTS / "01_mlflow_experiments.png").exists():
    story += fig(SHOTS / "01_mlflow_experiments.png", W - 3.2*cm, 6.5*cm,
                 "Screenshot 1 — MLflow Experiments Dashboard showing all training runs with metrics")
else:
    story.append(ph("📸 INSERT SCREENSHOT: MLflow Experiments UI\n(reports/screenshots/01_mlflow_experiments.png)", h=6*cm))

story.append(Spacer(1, 2*mm))
story.append(Paragraph(
    "The <b>model comparison report</b> is serialised to <code>reports/model_comparison.json</code> "
    "and the winning pipeline is saved as both <code>models/model_pipeline.joblib</code> and "
    "<code>models/heartguard_pipeline.joblib</code> for use by the FastAPI serving layer.",
    styles["body"]
))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 6: Task 4 — Model Packaging & Reproducibility
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("4 · Model Packaging & Reproducibility", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>4.1 Serialisation Formats</b>", styles["h2"]))
pack_rows = [
    ["Format", "File", "Purpose"],
    ["Joblib",    "models/model_pipeline.joblib",        "Primary serving format — fast, memory-mapped"],
    ["Joblib",    "models/heartguard_pipeline.joblib",   "Alias for API loading"],
    ["MLflow",    "models/mlflow_model/",                "Pyfunc bundle with input schema & env"],
    ["JSON",      "models/baseline_distribution.json",   "Pre-computed feature stats for drift detection"],
    ["JSON",      "models/model_metrics.json",           "Quality gate evaluation metrics"],
]
packt = Table(pack_rows, colWidths=[2.8*cm, 7*cm, 6.4*cm])
packt.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0), C_DARK),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 8),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_LIGHT, colors.white]),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
    ("TOPPADDING",  (0,0), (-1,-1), 3),
    ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("FONTNAME",    (1,1), (1,-1), "Courier"),
]))
story.append(packt)
story.append(Spacer(1, 4*mm))

story.append(Paragraph("<b>4.2 Environment Reproducibility</b>", styles["h2"]))
repr_items = [
    "<b>requirements.txt</b>: Fully pinned dependency manifest (scikit-learn, mlflow, fastapi, etc.).",
    "<b>environment.yml</b>: Conda environment file with Python 3.11 and exact package versions.",
    "<b>pyproject.toml</b>: Python package metadata, Ruff linting config, and pytest configuration.",
    "<b>src/config.py</b>: Centralised configuration with environment variable overrides — works identically on macOS, Linux, and Kubernetes.",
    "<b>Makefile / scripts/</b>: One-command pipeline execution from raw data to deployed API.",
]
for item in repr_items:
    story.append(Paragraph(f"• {item}", styles["bullet"]))
story.append(Spacer(1, 4*mm))

story.append(Paragraph("<b>4.3 Inference Engine</b> (<i>src/models/predict.py</i>)", styles["h2"]))
story.append(Paragraph(
    "The standalone inference engine loads the joblib pipeline and accepts either a single patient "
    "dict or a pandas DataFrame. It returns prediction label, probability scores, and latency. "
    "The FastAPI layer wraps this engine with Pydantic v2 physiological boundary validation "
    "(<i>schemas.py</i>).",
    styles["body"]
))

predict_code = """\
# Example prediction call (src/models/predict.py)
from src.models.predict import predict_patient
result = predict_patient({
    "age": 67.0, "sex": 1.0, "cp": 4.0, "trestbps": 160.0,
    "chol": 286.0, "fbs": 0.0, "restecg": 2.0, "thalach": 108.0,
    "exang": 1.0, "oldpeak": 1.5, "slope": 2.0, "ca": 3.0, "thal": 3.0
})
# => {"prediction": 1, "label": "Heart Disease Risk Detected",
#     "probability_disease": 0.994, "latency_ms": 1.2}"""
story.append(code_block(predict_code))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 7: Task 5 — CI/CD & Automated Testing
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("5 · CI/CD Pipeline & Automated Testing", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>5.1 Test Suite</b> (29 Tests, 100% Pass Rate)", styles["h2"]))
test_rows = [
    ["Test Module", "Tests", "Coverage Area"],
    ["test_data.py",     "5",  "Schema validation, cleaning, binary encoding, missing value detection"],
    ["test_features.py", "4",  "ColumnTransformer output dims, NaN elimination, train/test split stratification"],
    ["test_model.py",    "5",  "Pipeline prediction accuracy, calibration, false negative bounds"],
    ["test_drift.py",    "5",  "KS-test statistics, PSI calculation, drift threshold detection"],
    ["test_api.py",      "7",  "FastAPI /predict, /health, /metrics, /drift endpoints + batch inference"],
    ["test_config.py",   "3",  "Config path portability across environments"],
    ["TOTAL",           "29",  "100% pass rate · pytest -v"],
]
tt = Table(test_rows, colWidths=[4.5*cm, 1.5*cm, 10.2*cm])
tt.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0), C_DARK),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("BACKGROUND",  (0,-1), (-1,-1), HexColor("#E3F2FD")),
    ("FONTNAME",    (0,-1), (-1,-1), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 8),
    ("ROWBACKGROUNDS",(0,1),(-1,-2),[C_LIGHT, colors.white]),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
    ("TOPPADDING",  (0,0), (-1,-1), 3),
    ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("ALIGN",       (1,0), (1,-1), "CENTER"),
]))
story.append(tt)
story.append(Spacer(1, 4*mm))

story.append(Paragraph("<b>5.2 GitHub Actions CI/CD Workflow</b>", styles["h2"]))
story.append(Paragraph(
    "Two workflows are defined in <code>.github/workflows/</code>:",
    styles["body"]
))
ci_items = [
    "<b>ci.yml</b>: Triggered on every push/PR. Stages: dependency install → Ruff linting → pytest (29 tests) → smoke-test container build.",
    "<b>train.yml</b>: Retraining workflow with model quality gate — fails if test ROC-AUC < 0.85 or Recall < 0.80.",
]
for item in ci_items:
    story.append(Paragraph(f"• {item}", styles["bullet"]))
story.append(Spacer(1, 3*mm))

if (SHOTS / "06_github_actions_pipeline.png").exists():
    story += fig(SHOTS / "06_github_actions_pipeline.png", W - 3.2*cm, 5.8*cm,
                 "Screenshot 2 — GitHub Actions CI Pipeline (All Checks Passing)")
else:
    story.append(ph("📸 INSERT SCREENSHOT: GitHub Actions CI/CD Pipeline\n(reports/screenshots/06_github_actions_pipeline.png)", h=5.5*cm))

story.append(Spacer(1, 2*mm))

if (SHOTS / "03_fastapi_swagger_ui.png").exists():
    story += fig(SHOTS / "03_fastapi_swagger_ui.png", W - 3.2*cm, 5.0*cm,
                 "Screenshot 3 — FastAPI Swagger UI (/docs) with all endpoints")
else:
    story.append(ph("📸 INSERT SCREENSHOT: FastAPI Swagger UI\n(reports/screenshots/03_fastapi_swagger_ui.png)", h=4.5*cm))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 8: Task 6 — Docker Containerisation
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("6 · Model Containerisation", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>6.1 Dockerfile Design</b>", styles["h2"]))
docker_items = [
    "<b>Base image</b>: python:3.11-slim — minimal attack surface.",
    "<b>Non-root execution</b>: Application runs as unprivileged user <code>mlopsuser</code> (UID 1000) — container security hardening.",
    "<b>Embedded health check</b>: <code>HEALTHCHECK CMD curl -f http://localhost:8000/health</code> — K8s liveness compatible.",
    "<b>Efficient layer caching</b>: pip install before COPY of application code.",
    "<b>Exposed port</b>: 8000 — Uvicorn ASGI server with async lifespan model loading.",
]
for item in docker_items:
    story.append(Paragraph(f"• {item}", styles["bullet"]))
story.append(Spacer(1, 3*mm))

story.append(Paragraph("<b>6.2 API Endpoints</b>", styles["h2"]))
api_rows = [
    ["Method", "Endpoint", "Description"],
    ["GET",  "/",              "Service metadata, student ID, endpoint index"],
    ["GET",  "/health",        "Liveness & readiness probe (model status, uptime)"],
    ["GET",  "/metrics",       "Prometheus metrics scrape endpoint"],
    ["GET",  "/drift",         "Retrieve baseline distribution & drift thresholds"],
    ["POST", "/drift",         "Evaluate batch records for KS + PSI covariate drift"],
    ["POST", "/predict",       "Single patient risk prediction (JSON in/out)"],
    ["POST", "/predict/batch", "Batch prediction (array of patient records)"],
    ["GET",  "/app",           "Interactive clinical diagnostic web GUI"],
    ["GET",  "/docs",          "Swagger UI interactive API documentation"],
]
at = Table(api_rows, colWidths=[1.8*cm, 4.2*cm, 10.2*cm])
at.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0), C_DARK),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 8),
    ("ROWBACKGROUNDS",(0,1),(-1,-1),[C_LIGHT, colors.white]),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
    ("TOPPADDING",  (0,0), (-1,-1), 3),
    ("BOTTOMPADDING",(0,0),(-1,-1),3),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("FONTNAME",    (1,1), (1,-1), "Courier"),
    ("FONTNAME",    (0,1), (0,-1), "Helvetica-Bold"),
    ("TEXTCOLOR",   (0,1), (0,5), HexColor("#1565C0")),
    ("TEXTCOLOR",   (0,6), (0,-1), HexColor("#E65100")),
]))
story.append(at)
story.append(Spacer(1, 3*mm))

if (SHOTS / "02_docker_container_verification.png").exists():
    story += fig(SHOTS / "02_docker_container_verification.png", W - 3.2*cm, 5.5*cm,
                 "Screenshot 4 — Docker container live verification: health, predict, and drift endpoints")
else:
    story.append(ph("📸 INSERT SCREENSHOT: Docker Container Running + API Test\n(reports/screenshots/02_docker_container_verification.png)", h=5*cm))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 9: Task 7+8 — Production Deployment & Monitoring
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("7 · Production Kubernetes Deployment", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>7.1 Kubernetes Manifests</b> (<i>k8s/</i>)", styles["h2"]))
k8s_items = [
    "<b>Namespace</b>: <code>mlops</code> — isolated namespace for production resources.",
    "<b>Deployment</b>: 2 replicas of <code>heartguard-api:latest</code>, resource requests/limits, liveness & readiness probes.",
    "<b>Service (LoadBalancer)</b>: External access on port 8000.",
    "<b>Ingress</b>: Nginx ingress controller routing <code>/api/v1/</code> to the service.",
    "<b>HPA</b>: Horizontal Pod Autoscaler — scales 2–10 replicas at 70% CPU utilisation.",
    "<b>Helm Chart</b>: Full parameterised chart in <code>helm/heart-disease-api/</code> with values.yaml overrides.",
]
for item in k8s_items:
    story.append(Paragraph(f"• {item}", styles["bullet"]))
story.append(Spacer(1, 3*mm))

if (SHOTS / "04_kubernetes_deployment_status.png").exists():
    story += fig(SHOTS / "04_kubernetes_deployment_status.png", W - 3.2*cm, 5.5*cm,
                 "Screenshot 5 — Kubernetes cluster status: pods, services, HPA, and ingress")
else:
    story.append(ph("📸 INSERT SCREENSHOT: kubectl get all -n mlops\n(reports/screenshots/04_kubernetes_deployment_status.png)", h=5*cm))

story.append(Spacer(1, 2*mm))
story.append(Paragraph("8 · Monitoring & Logging", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>8.1 Observability Stack</b>", styles["h2"]))
mon_items = [
    "<b>Structured JSON Logging</b>: All API requests logged as JSON with correlation IDs (<i>src/api/middleware.py</i>).",
    "<b>Prometheus Metrics</b>: Request count, latency histograms, prediction outcomes scraped at <code>GET /metrics</code>.",
    "<b>Grafana Dashboard</b>: Pre-built dashboard (<code>monitoring/grafana_dashboard.json</code>) with request rate, latency P95, and prediction distribution panels.",
    "<b>Statistical Drift Engine</b> (<i>src/models/drift.py</i>): Real-time Kolmogorov-Smirnov test + Population Stability Index (PSI) for all 13 features via <code>POST /drift</code>.",
]
for item in mon_items:
    story.append(Paragraph(f"• {item}", styles["bullet"]))
story.append(Spacer(1, 2*mm))

if (SHOTS / "05_grafana_monitoring_dashboard.png").exists():
    story += fig(SHOTS / "05_grafana_monitoring_dashboard.png", W - 3.2*cm, 5.5*cm,
                 "Screenshot 6 — Grafana Monitoring Dashboard: request rate, latency, prediction distribution")
else:
    story.append(ph("📸 INSERT SCREENSHOT: Grafana Dashboard\n(reports/screenshots/05_grafana_monitoring_dashboard.png)", h=5*cm))

story.append(PageBreak())

# ══════════════════════════════════════════════════════════════════
# PAGE 10: Task 9 — Summary, Key Decisions & Checklist
# ══════════════════════════════════════════════════════════════════
story.append(Paragraph("9 · Summary, Design Decisions & Deliverables Checklist", styles["h1"]))
story.append(hr())

story.append(Paragraph("<b>9.1 Key Design Decisions</b>", styles["h2"]))
decisions = [
    "<b>Recall Priority</b>: Model selection score = 60% Recall + 40% ROC-AUC. False negatives in cardiac screening are life-threatening.",
    "<b>Zero-Leakage Pipeline</b>: All imputation and scaling inside CV folds — no test data contamination.",
    "<b>Clinical Interpretability</b>: Logistic Regression chosen for its log-odds transparency alongside superior AUC.",
    "<b>Non-Root Docker Security</b>: Container runs as <code>mlopsuser</code> UID 1000 — enterprise security standard.",
    "<b>Async Lifespan Model Loading</b>: FastAPI startup loads model once into shared application state — eliminates per-request IO.",
    "<b>Statistical Drift Detection</b>: KS test (p < 0.05) + PSI (> 0.2) dual-criterion ensures operationally robust covariate monitoring.",
]
for d in decisions:
    story.append(Paragraph(f"• {d}", styles["bullet"]))
story.append(Spacer(1, 2*mm))

story.append(Paragraph("<b>9.2 Assignment Deliverables Checklist</b>", styles["h2"]))
checklist_rows = [
    ["#", "Task", "Marks", "Status", "Key Artifacts"],
    ["1", "Data Acquisition & EDA",             "5",  "✓", "download_data.py · clean_data.py · reports/figures/*.png"],
    ["2", "Feature Engineering & Models",       "8",  "✓", "build_features.py · train.py · AUC=0.9632 · Recall=89.29%"],
    ["3", "MLflow Experiment Tracking",         "5",  "✓", "mlruns/ · model_comparison.json · experiment logs"],
    ["4", "Model Packaging & Reproducibility",  "7",  "✓", "model_pipeline.joblib · mlflow_model/ · requirements.txt"],
    ["5", "CI/CD & Automated Testing",          "8",  "✓", "29 pytest tests · ci.yml · train.yml · Ruff linting"],
    ["6", "Model Containerisation",             "5",  "✓", "Dockerfile (non-root) · /predict endpoint · test_docker.sh"],
    ["7", "Production Deployment",              "7",  "✓", "k8s/ manifests · helm/ chart · LoadBalancer · HPA · Ingress"],
    ["8", "Monitoring & Logging",               "3",  "✓", "Prometheus /metrics · Grafana · JSON logs · KS+PSI /drift"],
    ["9", "Documentation & Reporting",          "2",  "✓", "FINAL_REPORT.pdf · README.md · implementation-audit.md"],
    ["", "TOTAL", "50", "100%", "All deliverables verified & operational"],
]
cl = Table(checklist_rows, colWidths=[0.7*cm, 5*cm, 1.3*cm, 1.2*cm, 8.0*cm])
cl.setStyle(TableStyle([
    ("BACKGROUND",  (0,0), (-1,0), C_DARK),
    ("TEXTCOLOR",   (0,0), (-1,0), colors.white),
    ("FONTNAME",    (0,0), (-1,0), "Helvetica-Bold"),
    ("BACKGROUND",  (0,-1), (-1,-1), HexColor("#E3F2FD")),
    ("FONTNAME",    (0,-1), (-1,-1), "Helvetica-Bold"),
    ("FONTSIZE",    (0,0), (-1,-1), 7.5),
    ("ROWBACKGROUNDS",(0,1),(-1,-2),[C_LIGHT, colors.white]),
    ("GRID",        (0,0), (-1,-1), 0.3, C_LINE),
    ("TOPPADDING",  (0,0), (-1,-1), 2),
    ("BOTTOMPADDING",(0,0),(-1,-1),2),
    ("LEFTPADDING", (0,0), (-1,-1), 4),
    ("ALIGN",       (2,0), (3,-1), "CENTER"),
    ("TEXTCOLOR",   (3,1), (3,-2), HexColor("#2E7D32")),
    ("FONTNAME",    (3,1), (3,-2), "Helvetica-Bold"),
]))
story.append(cl)
story.append(Spacer(1, 3*mm))

story.append(Paragraph("<b>9.3 Setup & Run Instructions</b>", styles["h2"]))
setup_code = """\
# 1. Clone & setup
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt && pip install -e .

# 2. Download data & train
python src/data/download_data.py && python src/data/clean_data.py
python src/models/train.py

# 3. Launch API locally
uvicorn src.api.app:app --port 8000

# 4. Run tests
pytest -v                       # 29 tests, 100% pass rate

# 5. Docker
docker build -t heartguard-api:latest .
docker run -p 8000:8000 heartguard-api:latest

# 6. Kubernetes (Minikube)
minikube image load heartguard-api:latest
kubectl apply -f k8s/           # or: helm install heartguard ./helm/heart-disease-api

# 7. Full observability stack
docker compose up -d            # API + Prometheus (9090) + Grafana (3001)"""
story.append(code_block(setup_code))

story.append(Spacer(1, 1*mm))
story.append(hr(C_ACCENT))
story.append(Paragraph(
    "<b>Repository:</b> <font color='#1565C0'>https://github.com/ajitjena42/MLOps-Assignment-1</font>  "
    "   |   <b>Student:</b> Jena Ajit Anirudha Ranjita (2025AE05839)   |   <b>Course:</b> AIMLCZG523 MLOps — BITS Pilani",
    styles["meta"]
))

# ── Build PDF ────────────────────────────────────────────────────────────────
doc = SimpleDocTemplate(
    str(OUT_PDF),
    pagesize=A4,
    leftMargin=1.6*cm, rightMargin=1.6*cm,
    topMargin=1.8*cm, bottomMargin=1.8*cm,
    title="HeartGuard — MLOps Assignment 1 Report",
    author="Jena Ajit Anirudha Ranjita (2025AE05839)",
    subject="MLOps End-to-End Pipeline Report",
)

doc.build(story, canvasmaker=NumberedCanvas)
print(f"PDF saved → {OUT_PDF}")
print(f"Pages: ~10 (check rendered output)")
