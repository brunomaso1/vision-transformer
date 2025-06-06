import os
from pathlib import Path

from dotenv import load_dotenv, find_dotenv
from loguru import logger

from vision_transformer.utils import DatasetFormat

# Load environment variables from .env file if it exists
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Environment
ENVIRONMENT = os.getenv("ENVIRONMENT", "dev")
logger.info(f"Actual environment is: {ENVIRONMENT}")

# Folders
DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RAW_DATA_EXTRACTION_DIR = RAW_DATA_DIR / "EuroSAT_RGB"

# URLs
DATA_RAW_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

# FilesNames
DATA_RAW_FILE = "EuroSAT_RGB"
DATA_RAW_FILE_ZIP = f"{DATA_RAW_FILE}.zip"

# Datsets formats
DATASET_CONFIG = {
    DatasetFormat.YOLO: {
        "extension": ".txt",
        "interim_folderpath": INTERIM_DATA_DIR / f"{DATA_RAW_FILE}_yolo",
        "processed_folderpath": PROCESSED_DATA_DIR / f"{DATA_RAW_FILE}_yolo",
        "structure": "annotations",
        "classes_file": "classes.names",
    },
    DatasetFormat.HUGGINGFACE: {
        "extension": ".json",
        "interim_folderpath": INTERIM_DATA_DIR / f"{DATA_RAW_FILE}_huggingface",
        "processed_folderpath": PROCESSED_DATA_DIR / f"{DATA_RAW_FILE}_huggingface",
        "structure": "datasets",
    },
}

# Error handling
if not DATA_RAW_URL:
    raise ValueError("DATA_RAW_URL must be defined in the environment variables.")

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
