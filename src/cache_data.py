import requests
import json
from os import listdir, makedirs
from pathlib import Path
from tqdm import tqdm
from configuration import PDFS_PATH, CACHED_DATA_PATH, DOCUMENT_LAYOUT_ANALYSIS_URL


def save_cache_json(pdf_name, response):
    Path(CACHED_DATA_PATH, pdf_name.replace(".pdf", ".json")).write_text(json.dumps(response.json(), indent=4))


def cache_layout_analysis():
    for pdf_name in tqdm(sorted(listdir(PDFS_PATH))):
        if Path(CACHED_DATA_PATH, pdf_name.replace(".pdf", ".json")).exists():
            continue
        pdf_path: Path = Path(PDFS_PATH, pdf_name)
        with open(pdf_path, "rb") as pdf_file:
            files = {"file": pdf_file}
            response = requests.post(DOCUMENT_LAYOUT_ANALYSIS_URL, files=files)
            save_cache_json(pdf_name, response)


if __name__ == "__main__":
    cache_layout_analysis()
