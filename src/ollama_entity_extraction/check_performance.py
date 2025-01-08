import json
from os import listdir
from pathlib import Path
from time import time

from tqdm import tqdm

from configuration import ROOT_PATH
from ollama_entity_extraction.OllamaNameExtractor import OllamaNameExtractor
from ollama_entity_extraction.data_model.ConsoleTextColor import ConsoleTextColor
from ollama_entity_extraction.data_model.EntitiesDict import EntitiesDict

CACHED_DATA_PATH = Path(ROOT_PATH, "cejil_labeled_data/cached_data")


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
    json_path = Path(ROOT_PATH, f"cejil_labeled_data/ner_json_files/person_names_{ollama_model_name}.json")
    text_path = Path(ROOT_PATH, f"cejil_labeled_data/ner_text_files/person_names_{ollama_model_name}.txt")
    extractor.save_entities_dict(json_path, entities_dict)
    extractor.save_entity_texts(text_path, list(entities_dict.keys()))


def check_name_entity_performance(ollama_model_name: str):
    name_labels: list[str] = Path(ROOT_PATH, f"cejil_labeled_data/labels/name_labels.txt").read_text().split("\n")
    json_dict_path: Path = Path(ROOT_PATH, f"cejil_labeled_data/ner_json_files/person_names_{ollama_model_name}.json")
    if not json_dict_path.exists():
        # extract_and_save_all_name_entities(ollama_model_name)
        json_dict_path: Path = Path(ROOT_PATH, f"cejil_labeled_data/ner_json_files/person_names_.json")

    json_dict = json.loads(json_dict_path.read_text())

    # print(f"Total names: {len(json_dict.keys())}")
    # print(name_labels[-10:])

    entities_dict: EntitiesDict = EntitiesDict.from_dict(json_dict)
    ollama_name_extractor: OllamaNameExtractor = OllamaNameExtractor()
    start = time()
    entities_dict = ollama_name_extractor.find_unique_entities(entities_dict)
    finding_unique_entities_time = round(time() - start, 2)
    # ollama_name_extractor.print_formatted_entities(entities_dict)

    label_count = len(name_labels)
    false_positive_count = 0
    found_unique_names = list(entities_dict.keys())

    name_label_groups = []

    for name_group_string in name_labels:
        name_group = [n.strip() for n in name_group_string.split(",")]
        name_label_groups.append(name_group)

    found_name_group_indexes = []

    for name in found_unique_names:
        name_found = False
        for name_group_index, name_label_group in enumerate(name_label_groups):
            if name_group_index in found_name_group_indexes:
                continue
            for name_label in name_label_group:
                if name == name_label:
                    found_name_group_indexes.append(name_group_index)
                    name_found = True
                    break
            if name_found:
                break
        if not name_found:
            false_positive_count += 1

    found_label_count = len(found_name_group_indexes)
    not_found_label_count = label_count - found_label_count

    print(f"\n{ConsoleTextColor.BLUE.value}")
    print(f"Model: {ollama_model_name}\n")
    print(f"Labels: {label_count}")
    print(f"Found labels: {found_label_count}")
    print(f"False positives: {false_positive_count}")
    print(f"Not found labels: {not_found_label_count}")
    print(f"Accuracy: {found_label_count / (label_count + false_positive_count) * 100:.2f}%")
    print(f"Recall: {found_label_count / (found_label_count + not_found_label_count) * 100:.2f}%")
    print(f"Precision: {found_label_count / (found_label_count + false_positive_count) * 100:.2f}%")
    print(f"Finding unique entities finished in {finding_unique_entities_time} seconds")
    print(f"{ConsoleTextColor.END.value}")




if __name__ == '__main__':
    # extract_and_save_all_name_entities("llama3.1")
    # check_name_entity_performance("llama3.1")
    check_name_entity_performance("nemotron-mini:latest")
