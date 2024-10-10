from os import makedirs
from pathlib import Path

SRC_PATH = Path(__file__).parent.absolute()
ROOT_PATH = Path(__file__).parent.parent.absolute()
CACHED_DATA_PATH = Path(ROOT_PATH, "data", "cached_data")
PDFS_PATH = Path(ROOT_PATH, "data", "pdfs")
VISUALIZATIONS_PATH = Path(ROOT_PATH, "data", "visualizations")
DOCUMENT_LAYOUT_ANALYSIS_URL = "http://localhost:5060"

if not CACHED_DATA_PATH.parent.exists():
    makedirs(CACHED_DATA_PATH.parent, exist_ok=True)
if not CACHED_DATA_PATH.exists():
    makedirs(CACHED_DATA_PATH, exist_ok=True)
if not PDFS_PATH.exists():
    makedirs(PDFS_PATH, exist_ok=True)
if not VISUALIZATIONS_PATH.exists():
    makedirs(VISUALIZATIONS_PATH, exist_ok=True)
if not Path(ROOT_PATH, "lol").exists():
    makedirs(Path(ROOT_PATH, "lol"), exist_ok=True)

if __name__ == "__main__":
    print(SRC_PATH)
    print(ROOT_PATH)
