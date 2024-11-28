import json
from os import listdir
from pathlib import Path
from tqdm import tqdm
from configuration import ROOT_PATH
from ollama_entity_extraction.OllamaLocationExtractor import OllamaLocationExtractor
from ollama_entity_extraction.OllamaNameExtractor import OllamaNameExtractor
from ollama_entity_extraction.OllamaOrganizationExtractor import OllamaOrganizationExtractor
from ollama_entity_extraction.data_model.EntitiesDict import EntitiesDict

CACHED_DATA_PATH = Path(ROOT_PATH, "cejil_data/cached_data")


def extract_and_save_all_name_entities(ollama_model_name: str):
    extractor = OllamaNameExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
    entities_dict.sort_entities()
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/person_names_{ollama_model_name}.json")
    text_path = Path(ROOT_PATH, f"cejil_data/ner_text_files/person_names_{ollama_model_name}.txt")
    extractor.save_entities_dict(json_path, entities_dict)
    extractor.save_entity_texts(text_path, list(entities_dict.keys()))


def save_sub_name_entities(ollama_model_name: str, entity_count: int = 210):
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/person_names_{ollama_model_name}.json")
    sub_names = list(json.loads(json_path.read_text()).keys())[:entity_count]
    sub_json_data = {k: json.loads(json_path.read_text())[k] for k in sub_names}
    sub_json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/sub_person_names_{ollama_model_name}.json")
    sub_text_path = Path(ROOT_PATH, f"cejil_data/ner_text_files/sub_person_names_{ollama_model_name}.txt")
    OllamaNameExtractor.save_entities_dict(sub_json_path, EntitiesDict.from_dict(sub_json_data))
    OllamaNameExtractor.save_entity_texts(sub_text_path, sub_names)


def run_unique_name_extraction_on_sub_data(ollama_model_name: str):
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/sub_person_names_{ollama_model_name}.json")
    if not json_path.exists():
        save_sub_name_entities(ollama_model_name)
    entities_dict = EntitiesDict.from_dict(json.loads(json_path.read_text()))
    extractor = OllamaNameExtractor()
    entities_dict = extractor.find_unique_entities(entities_dict)
    extractor.print_formatted_entities(entities_dict)


def run_unique_name_extraction_example():
    extractor = OllamaNameExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
        break
    entities_dict = extractor.find_unique_entities(entities_dict)
    OllamaNameExtractor.print_formatted_entities(entities_dict)


def extract_and_save_all_organization_entities(ollama_model_name: str):
    extractor = OllamaOrganizationExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
    entities_dict.sort_entities()
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/organization_names_{ollama_model_name}.json")
    text_path = Path(ROOT_PATH, f"cejil_data/ner_text_files/organization_names_{ollama_model_name}.txt")
    extractor.save_entities_dict(json_path, entities_dict)
    extractor.save_entity_texts(text_path, list(entities_dict.keys()))


def save_sub_organization_entities(ollama_model_name: str, entity_count: int = 210):
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/organization_names_{ollama_model_name}.json")
    sub_names = list(json.loads(json_path.read_text()).keys())[:entity_count]
    sub_json_data = {k: json.loads(json_path.read_text())[k] for k in sub_names}
    sub_json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/sub_organization_names_{ollama_model_name}.json")
    sub_text_path = Path(ROOT_PATH, f"cejil_data/ner_text_files/sub_organization_names_{ollama_model_name}.txt")
    OllamaOrganizationExtractor.save_entities_dict(sub_json_path, EntitiesDict.from_dict(sub_json_data))
    OllamaOrganizationExtractor.save_entity_texts(sub_text_path, sub_names)


def run_unique_organization_extraction_on_sub_data(ollama_model_name: str):
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/sub_organization_names_{ollama_model_name}.json")
    if not json_path.exists():
        save_sub_organization_entities(ollama_model_name)
    entities_dict = EntitiesDict.from_dict(json.loads(json_path.read_text()))
    extractor = OllamaNameExtractor()
    entities_dict = extractor.find_unique_entities(entities_dict)
    extractor.print_formatted_entities(entities_dict)


def run_unique_organization_extraction_example():
    extractor = OllamaOrganizationExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
        break
    entities_dict = extractor.find_unique_entities(entities_dict)
    extractor.print_formatted_entities(entities_dict)


def extract_and_save_all_gpe_entities(ollama_model_name: str):
    extractor = OllamaLocationExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
    entities_dict.sort_entities()
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/gpe_entities_{ollama_model_name}.json")
    text_path = Path(ROOT_PATH, f"cejil_data/ner_text_files/gpe_entities_{ollama_model_name}.txt")
    extractor.save_entities_dict(json_path, entities_dict)
    extractor.save_entity_texts(text_path, list(entities_dict.keys()))


def save_sub_gpe_entities(ollama_model_name: str, entity_count: int = 210):
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/gpe_entities_{ollama_model_name}.json")
    sub_names = list(json.loads(json_path.read_text()).keys())[:entity_count]
    sub_json_data = {k: json.loads(json_path.read_text())[k] for k in sub_names}
    sub_json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/sub_gpe_entities_{ollama_model_name}.json")
    sub_text_path = Path(ROOT_PATH, f"cejil_data/ner_text_files/sub_gpe_entities_{ollama_model_name}.txt")
    OllamaLocationExtractor.save_entities_dict(sub_json_path, EntitiesDict.from_dict(sub_json_data))
    OllamaLocationExtractor.save_entity_texts(sub_text_path, sub_names)


def run_unique_gpe_extraction_on_sub_data(ollama_model_name: str):
    json_path = Path(ROOT_PATH, f"cejil_data/ner_json_files/sub_gpe_entities_{ollama_model_name}.json")
    if not json_path.exists():
        save_sub_gpe_entities(ollama_model_name)
    entities_dict = EntitiesDict.from_dict(json.loads(json_path.read_text()))
    extractor = OllamaLocationExtractor()
    entities_dict = extractor.find_unique_entities(entities_dict)
    # extractor.print_formatted_entities(entities_dict)


def run_unique_gpe_extraction_example():
    extractor = OllamaLocationExtractor()
    entities_dict: EntitiesDict = EntitiesDict()
    for file in tqdm(sorted(listdir(CACHED_DATA_PATH))):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        extractor.extract_entities(pdf_name, segment_boxes, entities_dict)
        break
    entities_dict = extractor.find_unique_entities(entities_dict)
    extractor.print_formatted_entities(entities_dict)


if __name__ == "__main__":
    # extract_and_save_all_gpe_entities("llama3.1")
    # save_sub_gpe_entities("llama3.1")
    run_unique_gpe_extraction_on_sub_data("llama3.1")
