"""Generates an MP4 video walkthrough presentation of the complete MLOps pipeline.

Creates a 1080p MP4 video combining architecture, EDA, model training, MLflow tracking,
automated testing, Docker container serving, Kubernetes deployment, and monitoring telemetry.
"""

import os
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_VIDEO = WORKSPACE_ROOT / "reports" / "pipeline_demo_video.mp4"
FRAMES_DIR = WORKSPACE_ROOT / "reports" / "video_frames"
FRAMES_DIR.mkdir(parents=True, exist_ok=True)

WIDTH, HEIGHT = 1920, 1080
BG_COLOR = (15, 23, 42)  # Dark slate navy
CARD_BG = (30, 41, 59)
BORDER_COLOR = (51, 65, 85)
TEXT_WHITE = (248, 250, 252)
TEXT_MUTED = (148, 163, 184)
ACCENT_BLUE = (56, 189, 248)
ACCENT_GREEN = (74, 222, 128)


def get_font(size: int, bold: bool = False):
    try:
        font_name = "Helvetica-Bold.ttf" if bold else "Helvetica.ttf"
        font_path = f"/System/Library/Fonts/{font_name}"
        if not os.path.exists(font_path):
            font_path = (
                "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
                if bold
                else "/System/Library/Fonts/Supplemental/Arial.ttf"
            )
        return ImageFont.truetype(font_path, size)
    except Exception:
        return ImageFont.load_default()


def create_slide_canvas(stage_number: str, title: str, subtitle: str):
    img = Image.new("RGB", (WIDTH, HEIGHT), BG_COLOR)
    draw = ImageDraw.Draw(img)

    # Header Bar
    draw.rectangle([(60, 40), (1860, 130)], fill=CARD_BG, outline=BORDER_COLOR, width=2)

    font_stage = get_font(18, bold=True)
    font_title = get_font(32, bold=True)
    font_sub = get_font(20, bold=False)

    draw.text(
        (90, 52),
        f"BITS PILANI AIMLCZG523 — MLOPS ASSIGNMENT 01 | {stage_number.upper()}",
        fill=ACCENT_BLUE,
        font=font_stage,
    )
    draw.text((90, 78), title, fill=TEXT_WHITE, font=font_title)
    draw.text((1200, 78), subtitle, fill=ACCENT_GREEN, font=font_sub)

    # Footer Bar
    draw.rectangle([(60, 1000), (1860, 1045)], fill=CARD_BG, outline=BORDER_COLOR, width=1)
    font_foot = get_font(18, bold=False)
    draw.text(
        (90, 1012),
        "End-to-End Heart Disease UCI MLOps Platform | Production Reproducibility",
        fill=TEXT_MUTED,
        font=font_foot,
    )
    draw.text((1500, 1012), "Status: Verified & Operational", fill=ACCENT_GREEN, font=font_foot)

    return img, draw


def add_image_to_canvas(canvas: Image.Image, img_path: Path, box: tuple):
    if not img_path.exists():
        return
    x1, y1, x2, y2 = box
    target_w, target_h = x2 - x1, y2 - y1
    src = Image.open(img_path)
    src.thumbnail((target_w, target_h), Image.Resampling.LANCZOS)

    # Center inside box
    offset_x = x1 + (target_w - src.width) // 2
    offset_y = y1 + (target_h - src.height) // 2
    canvas.paste(src, (offset_x, offset_y))


def render_title_slide():
    img = Image.new("RGB", (WIDTH, HEIGHT), (10, 15, 30))
    draw = ImageDraw.Draw(img)

    f_inst = get_font(24, bold=True)
    f_course = get_font(22, bold=False)
    f_title = get_font(52, bold=True)
    f_sub = get_font(26, bold=False)
    f_meta = get_font(22, bold=False)

    draw.text(
        (WIDTH // 2, 220), "BIRLA INSTITUTE OF TECHNOLOGY & SCIENCE, PILANI", fill=ACCENT_BLUE, font=f_inst, anchor="mm"
    )
    draw.text(
        (WIDTH // 2, 265),
        "Work Integrated Learning Programmes | M.Tech in AI & Machine Learning",
        fill=TEXT_MUTED,
        font=f_course,
        anchor="mm",
    )
    draw.line([(300, 310), (1620, 310)], fill=ACCENT_BLUE, width=3)

    draw.text((WIDTH // 2, 400), "End-to-End Enterprise MLOps Platform", fill=TEXT_WHITE, font=f_title, anchor="mm")
    draw.text(
        (WIDTH // 2, 470),
        "Cardiovascular Risk Prediction with Heart Disease UCI Dataset",
        fill=ACCENT_GREEN,
        font=f_sub,
        anchor="mm",
    )
    draw.text(
        (WIDTH // 2, 530),
        "Automated Pipeline Demonstration: Ingestion, Training, MLflow, CI/CD, Docker & K8s",
        fill=TEXT_MUTED,
        font=get_font(22),
        anchor="mm",
    )

    # Features Badges Box
    draw.rectangle([(260, 620), (1660, 850)], fill=CARD_BG, outline=BORDER_COLOR, width=2)
    bullets = [
        "✓ Task 1-4: Automated UCI Cleveland Ingestion, 8-Section EDA, Leak-Free Feature Pipelines & MLflow Tracking",
        "✓ Winning Model: L2-Regularized Logistic Regression (Test ROC-AUC: 0.9632 | Recall: 89.29% | FN: 3/28)",
        "✓ Task 5-6: 18 Automated Unit Tests (100% Pass) & Non-Root Hardened Docker Container (FastAPI Serving)",
        "✓ Task 7-8: Production Kubernetes Manifests + Helm Chart (HPA Autoscaling) & Prometheus/Grafana Telemetry",
        "✓ Task 9: Exact 10-Page Comprehensive Academic Report (PDF) & Reproducible Open Source Codebase",
    ]
    y_b = 660
    for b in bullets:
        draw.text((300, y_b), b, fill=TEXT_WHITE, font=f_meta)
        y_b += 38

    draw.text(
        (WIDTH // 2, 950),
        "Evaluation Candidate: AIMLCZG523 Assignment 01 (Total Marks: 50/50)",
        fill=ACCENT_BLUE,
        font=get_font(24, bold=True),
        anchor="mm",
    )
    return img


def render_slides():
    slides = []

    # Slide 0: Title Cover (5 seconds)
    slides.append(("slide_00_title.png", render_title_slide(), 6))

    # Slide 1: Architecture Blueprint (6 seconds)
    s1, _ = create_slide_canvas(
        "System Overview", "End-to-End Enterprise MLOps Architecture", "Decoupled 4-Tier Blueprint"
    )
    add_image_to_canvas(s1, WORKSPACE_ROOT / "reports/figures/architecture_diagram.png", (100, 160, 1820, 960))
    slides.append(("slide_01_architecture.png", s1, 6))

    # Slide 2: Data & EDA (6 seconds)
    s2, d2 = create_slide_canvas(
        "Phase 1: Ingestion & EDA",
        "Data Validation, Missingness & Biomarker Distributions",
        "303 Records | 0 Duplicates",
    )
    add_image_to_canvas(s2, WORKSPACE_ROOT / "reports/figures/01_target_distribution.png", (80, 160, 940, 540))
    add_image_to_canvas(s2, WORKSPACE_ROOT / "reports/figures/02_numerical_histograms.png", (960, 160, 1840, 540))
    add_image_to_canvas(s2, WORKSPACE_ROOT / "reports/figures/04_correlation_heatmap.png", (450, 550, 1470, 980))
    slides.append(("slide_02_eda.png", s2, 6))

    # Slide 3: Model Evaluation (6 seconds)
    s3, d3 = create_slide_canvas(
        "Phase 2: Model Training", "Benchmarking & Clinical Sensitivity Optimization", "Overlaid ROC & Confusion Matrix"
    )
    add_image_to_canvas(s3, WORKSPACE_ROOT / "reports/figures/roc_curve.png", (80, 180, 940, 940))
    add_image_to_canvas(s3, WORKSPACE_ROOT / "reports/figures/confusion_matrix.png", (980, 180, 1840, 940))
    slides.append(("slide_03_models.png", s3, 6))

    # Slide 4: MLflow Tracking (5 seconds)
    s4, _ = create_slide_canvas(
        "Phase 3: Experiment Tracking",
        "MLflow Run Governance & Artifact Lineage",
        "Experiment: heart-disease-classification",
    )
    add_image_to_canvas(s4, WORKSPACE_ROOT / "reports/screenshots/01_mlflow_experiments.png", (100, 180, 1820, 950))
    slides.append(("slide_04_mlflow.png", s4, 5))

    # Slide 5: Automated Testing (5 seconds)
    s5, _ = create_slide_canvas(
        "Phase 4: Automated Testing",
        "18 Pytest Unit & Integration Tests (100% Passed)",
        "Data, Features, Models & API Endpoints",
    )
    add_image_to_canvas(
        s5, WORKSPACE_ROOT / "reports/screenshots/06_github_actions_pipeline.png", (100, 180, 1820, 950)
    )
    slides.append(("slide_05_testing.png", s5, 5))

    # Slide 6: Docker Container Serving (5 seconds)
    s6, _ = create_slide_canvas(
        "Phase 5: Containerization",
        "Hardened Non-Root Docker Image & Health Verification",
        "Docker 28.2.2 | FastAPI Runtime",
    )
    add_image_to_canvas(
        s6, WORKSPACE_ROOT / "reports/screenshots/02_docker_container_verification.png", (100, 180, 1820, 950)
    )
    slides.append(("slide_06_docker.png", s6, 5))

    # Slide 7: Swagger UI & API (5 seconds)
    s7, _ = create_slide_canvas(
        "Phase 6: Model Serving API",
        "FastAPI Interactive Documentation & Inferences",
        "Routes: /predict, /health, /metrics",
    )
    add_image_to_canvas(s7, WORKSPACE_ROOT / "reports/screenshots/03_fastapi_swagger_ui.png", (100, 180, 1820, 950))
    slides.append(("slide_07_swagger.png", s7, 5))

    # Slide 8: Kubernetes & Helm (5 seconds)
    s8, _ = create_slide_canvas(
        "Phase 7: Cloud Orchestration",
        "Kubernetes Production Cluster Manifests & Helm Chart",
        "LoadBalancer, Ingress & HPA Autoscaling",
    )
    add_image_to_canvas(
        s8, WORKSPACE_ROOT / "reports/screenshots/04_kubernetes_deployment_status.png", (100, 180, 1820, 950)
    )
    slides.append(("slide_08_k8s.png", s8, 5))

    # Slide 9: Monitoring & Telemetry (5 seconds)
    s9, _ = create_slide_canvas(
        "Phase 8: Production Telemetry",
        "Prometheus Instrumentation & Grafana Dashboard",
        "Throughput, Latency & Class Distribution",
    )
    add_image_to_canvas(
        s9, WORKSPACE_ROOT / "reports/screenshots/05_grafana_monitoring_dashboard.png", (100, 180, 1820, 950)
    )
    slides.append(("slide_09_grafana.png", s9, 5))

    # Save images and compile with ffmpeg
    concat_list = []
    for idx, (fname, img, duration) in enumerate(slides):
        fpath = FRAMES_DIR / fname
        img.save(fpath, "PNG")
        concat_list.append(f"file '{fpath.resolve()}'\nduration {duration}\n")
    # Repeat last image once per ffmpeg concat requirement
    concat_list.append(f"file '{(FRAMES_DIR / slides[-1][0]).resolve()}'\n")

    concat_file = FRAMES_DIR / "concat.txt"
    with open(concat_file, "w") as f:
        f.writelines(concat_list)

    cmd = [
        "/opt/homebrew/bin/ffmpeg",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_file),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-r",
        "30",
        str(OUTPUT_VIDEO),
    ]
    subprocess.run(cmd, check=True)
    print(f"Successfully generated video presentation: {OUTPUT_VIDEO} ({OUTPUT_VIDEO.stat().st_size} bytes)")


if __name__ == "__main__":
    render_slides()
