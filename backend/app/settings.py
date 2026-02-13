import os


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")

# Default storage for PDFs and temp images (override via env vars if needed)
PDF_FOLDER = os.environ.get("PDF_FOLDER", os.path.join(DATA_DIR, "pdfs"))
TEMP_IMAGE_FOLDER = os.environ.get("TEMP_IMAGE_FOLDER", os.path.join(DATA_DIR, "temp_images"))

# Tesseract path (override via env var)
TESSERACT_CMD = os.environ.get(
    "TESSERACT_CMD",
    r"C:\LibrerieAggiuntePython\Tesseract-OCR\tesseract.exe",
)


OPENAI_MODEL = "gpt-5.2"


def ensure_dirs() -> None:
    os.makedirs(PDF_FOLDER, exist_ok=True)
    os.makedirs(TEMP_IMAGE_FOLDER, exist_ok=True)
