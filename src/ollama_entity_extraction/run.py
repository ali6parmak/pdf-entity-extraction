import json
from os import listdir
from pathlib import Path
from tqdm import tqdm
from configuration import ROOT_PATH
from ollama_entity_extraction.OllamaNameExtractor import OllamaNameExtractor
from ollama_entity_extraction.ollama_temp import EntitiesDict

CACHED_DATA_PATH = Path(ROOT_PATH, "cejil_data/cached_data")


def extract_and_save_name_entities(ollama_model_name: str):
    extractor = OllamaNameExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
    json_path = Path(ROOT_PATH, "cejil_data/ner_json_files/person_names_{ollama_model_name}.json")
    text_path = Path(ROOT_PATH, "cejil_data/ner_text_files/person_names_{ollama_model_name}.txt")
    extractor.save_entities_dict(json_path, entities_dict)
    entity_texts = entities_dict.keys()
    extractor.save_entity_texts(text_path, entity_texts)


def run_name_extraction_example():
    extractor = OllamaNameExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
        break
    OllamaNameExtractor.print_formatted_entities(entities_dict)


if __name__ == "__main__":
    run_name_extraction_example()
