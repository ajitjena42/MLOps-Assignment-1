"""Data acquisition module for UCI Heart Disease Dataset (Cleveland variant).

Source:
    UCI Machine Learning Repository
    Creators:
        1. Hungarian Institute of Cardiology. Budapest: Andras Janosi, M.D.
        2. University Hospital, Zurich, Switzerland: William Steinbrunn, M.D.
        3. University Hospital, Basel, Switzerland: Matthias Pfisterer, M.D.
        4. V.A. Medical Center, Long Beach and Cleveland Clinic Foundation:
           Robert Detrano, M.D., Ph.D.
    Donor: David W. Aha (aha@ics.uci.edu)
    License: Creative Commons Attribution 4.0 International (CC BY 4.0)
    URL: https://archive.ics.uci.edu/dataset/45/heart+disease
"""

import logging
import sys
import urllib.request
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("download_data")

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
FALLBACK_URL = "https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/heart.csv"

# Official 14 clinical attributes used in machine learning research
COLUMN_NAMES = [
    "age",       # Age in years
    "sex",       # Sex (1 = male, 0 = female)
    "cp",        # Chest pain type (1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic)
    "trestbps",  # Resting blood pressure in mm Hg
    "chol",      # Serum cholesterol in mg/dl
    "fbs",       # Fasting blood sugar > 120 mg/dl (1 = true, 0 = false)
    "restecg",   # Resting electrocardiographic results (0, 1, 2)
    "thalach",   # Maximum heart rate achieved
    "exang",     # Exercise induced angina (1 = yes, 0 = no)
    "oldpeak",   # ST depression induced by exercise relative to rest
    "slope",     # Slope of peak exercise ST segment (1: upsloping, 2: flat, 3: downsloping)
    "ca",        # Number of major vessels (0-3) colored by fluoroscopy
    "thal",      # Thalassemia (3 = normal, 6 = fixed defect, 7 = reversible defect)
    "target"     # Heart disease diagnosis (0: absence, 1-4: presence)
]


def download_dataset(output_path: str = "data/raw/heart.csv", force: bool = False) -> Path:
    """Download the raw UCI Heart Disease dataset.
    
    Args:
        output_path: Destination path for the raw CSV.
        force: If True, redownload even if file exists.
        
    Returns:
        Path to the downloaded file.
    """
    dest = Path(output_path)
    if dest.exists() and not force:
        logger.info(f"Dataset already exists at {dest}. Skipping download.")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"Attempting to download dataset from primary UCI URL: {UCI_URL}")

    download_success = False
    temp_file = dest.with_suffix(".tmp")

    # 1. Try primary UCI URL
    try:
        urllib.request.urlretrieve(UCI_URL, temp_file)
        # Verify lines
        df = pd.read_csv(temp_file, header=None, names=COLUMN_NAMES, na_values="?")
        if len(df) >= 250 and len(df.columns) == 14:
            df.to_csv(dest, index=False)
            download_success = True
            logger.info(f"Successfully downloaded and verified {len(df)} records from UCI Repository.")
    except Exception as e:
        logger.warning(f"Failed to fetch from primary UCI URL ({e}). Attempting secondary mirror...")

    # 2. Try secondary fallback if primary failed
    if not download_success:
        try:
            urllib.request.urlretrieve(FALLBACK_URL, temp_file)
            df = pd.read_csv(temp_file)
            # Standardize column naming if necessary
            rename_map = {col: col.lower().strip() for col in df.columns}
            df = df.rename(columns=rename_map)
            if "target" not in df.columns and "num" in df.columns:
                df = df.rename(columns={"num": "target"})
            elif "target" not in df.columns and "condition" in df.columns:
                df = df.rename(columns={"condition": "target"})
            df.to_csv(dest, index=False)
            download_success = True
            logger.info(f"Successfully downloaded {len(df)} records from fallback repository.")
        except Exception as e:
            logger.error(f"Fallback download also failed: {e}")

    # Clean up temp file
    if temp_file.exists():
        temp_file.unlink()

    if not download_success:
        raise RuntimeError(
            f"Could not download Heart Disease dataset from either {UCI_URL} or {FALLBACK_URL}. "
            f"Please verify internet connectivity or manually download 'processed.cleveland.data' "
            f"and place it at {dest}."
        )

    return dest


if __name__ == "__main__":
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "data/raw/heart.csv"
    download_dataset(out_dir)
