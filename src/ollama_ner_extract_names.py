import json
from collections import defaultdict
from os import listdir
from pathlib import Path
import pickle
from time import time

from Levenshtein import ratio
from flair.data import Sentence
from flair.nn import Classifier
from ollama import Client

CACHED_DATA_PATH = Path("./projects/pdf-entity-extraction/cejil_data/cached_data")

# MODEL_NAME = "qwen2.5:32b"
# MODEL_NAME = "gemma2:27b"
MODEL_NAME = "llama3.1"
# MODEL_NAME = "aya:35b"
HOST = "http://localhost:11434"

# MODEL_NAME = "gemma2:27b"
# HOST = "http://34.147.103.238:11434"

# model load times not included
# qwen2.5:32b 36.88 seconds
# aya:35b - 277.8 seconds
# gemma2:27b - 30.24 seconds
# llama3.1 - 16.18 seconds (run on local - gtx 1070)

def convert_sets_to_lists(data):
  if isinstance(data, dict):
      return {k: convert_sets_to_lists(v) for k, v in data.items()}
  elif isinstance(data, set):
      return list(data)
  elif isinstance(data, list):
      return [convert_sets_to_lists(element) for element in data]
  elif isinstance(data, tuple):
      return tuple(convert_sets_to_lists(element) for element in data)
  else:
      return data


def save_to_json(entities_dict):
    json_compatible_data = convert_sets_to_lists(entities_dict)
    output_path = Path("./projects/pdf-entity-extraction/cejil_data/ner_json_files/person_names.json")
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(json_compatible_data, f, ensure_ascii=False, indent=4)

def save_to_pickle(entities_dict):
    output_path = Path("./projects/pdf-entity-extraction/cejil_data/ner_pickle_files/person_names.pickle")
    with output_path.open('wb') as f:
        pickle.dump(entities_dict, f)

def save_to_text_file():
    json_data = json.loads(Path("./projects/pdf-entity-extraction/cejil_data/ner_json_files/person_names.json").read_text())
    names_list: list[str] = []
    for name in list(json_data.keys()):
        names_list.append(name)
    names_list = list(set(names_list))
    names_list.sort()
    output_path = Path("./projects/pdf-entity-extraction/cejil_data/ner_text_files/person_names.txt")
    output_path.write_text("\n".join(names_list))

def extract_and_save_entities():
    extractor = OllamaMultipleEntityExtractor()
    entities_dict = {}
    for file in sorted(listdir(CACHED_DATA_PATH)):
        if ".json" not in file:
            continue
        pdf_name = file.replace(".json", "")
        extractor.extract_entities(pdf_name, entities_dict)
    save_to_pickle(entities_dict)
    save_to_json(entities_dict)
    save_to_text_file()


class OllamaMultipleEntityExtractor:
    def __init__(self):
        self.flair_model = Classifier.load("ner-ontonotes-large")

    @staticmethod
    def get_person_entity_prompt(entity_texts: list[str]):
        entities_string = "\n".join(entity_texts)
        content = (
            "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of person names "
            "and consolidate entries that refer to the same individual, even if there are minor variations due to typos, accents, or missing names."
            "\n\n"
            "Here are the instructions that you should follow:\n"
            "\n - Identify names that refer to the same person."
            "\n - Consider variations such as spelling differences, missing middle names, or presence of accents (e.g., \"Ali\" vs. \"Alí\")."
            "\n - For each group of similar names, choose the most complete and accurate version as the representative name."
            # "\n - Prefer names that include full middle names over those that have only initial names or missing middle names."
            "\n - Prefer names that include full middle names over those that have initials or omit them."
            "\n - Do not skip any entities in the output, return all the unique entities."
            "\n - Do not create new, additional entities, just output the consolidated list."
            "\n - Do not make any additional explanations, just output the entities."
            "\n\n\nHere are the names to process:\n\n"
            "### INPUT\n\n"
            f"{entities_string}\n\n\n"
            f"### OUTPUT:")
        return content

    @staticmethod
    def remove_overlapping_entities(entities, start_key="start_index", end_key="end_index"):
        sorted_entities = sorted(entities, key=lambda x: (x[start_key], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity[start_key] >= last_end:
                result.append(entity)
                last_end = entity[end_key]

        return result

    def extract_entities_from_text(self, text: str):
        entities = []
        entity_results = []

        sentence = Sentence(text)
        self.flair_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag not in {"PERSON"}:
                continue
            entity_results.append(
                {
                    "text": entity.text,
                    "entity_label": entity.tag,
                    "start_index": entity.start_position,
                    "end_index": entity.end_position,
                }
            )

        entity_results = self.remove_overlapping_entities(entity_results)
        for entity in entity_results:
            entities.append([entity["text"], entity["entity_label"]])
        return entities

    def extract_entities(self, pdf_name: str, entities_dict: dict = None):
        def create_default_dict():
            # return {"pages": set(), "mentions": set(), "pdf_name": list()}
            # return {"pages": set(), "mentions": set(), "mention_starts": list(), "mention_ends": list()}
            return {"pages": list(), "mentions": list(), "mention_starts": list(), "mention_ends": list(), "segment_numbers": list()}

        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        previous_pages_segment_count: int = 0
        current_page = 1
        segment_no = 1

        for segment_box in segment_boxes:
            if segment_box["page_number"] != current_page:
                previous_pages_segment_count += len([segment_box for segment_box in segment_boxes if segment_box["page_number"] == current_page])
                current_page = segment_box["page_number"]
                segment_no = 1
            reconstructed_text = " ".join([word for word in segment_box["text"].split()])
            if not reconstructed_text:
                continue
            segment_entities = self.extract_entities_from_text(reconstructed_text)
            for entity_text, entity_label in segment_entities:
                entity_text_title = entity_text.title()
                # if entity_label not in entities_dict:
                #     entities_dict[entity_label] = {}
                if entity_text_title not in entities_dict:
                    entities_dict[entity_text_title] = create_default_dict()

                entities_dict[entity_text_title]["pages"].append(pdf_name + " - p:" + str(segment_box["page_number"]))
                # entities_dict[entity_text_title]["mentions"].add(reconstructed_text)
                entities_dict[entity_text_title]["mentions"].append(reconstructed_text)
                entities_dict[entity_text_title]["mention_starts"].append(reconstructed_text.index(entity_text))
                entities_dict[entity_text_title]["mention_ends"].append(reconstructed_text.index(entity_text) + len(entity_text))
                entities_dict[entity_text_title]["segment_numbers"].append(segment_no)
                # entities_dict[entity_label][entity_text_title]["pdf_names"].append(pdf_name)
            segment_no += 1


        return entities_dict

    @staticmethod
    def print_formatted_entities(entities_dict):
        PINK = '\033[95m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

        # Example usage
        print(f"{BOLD}This is bold text{END}")
        print(f"{UNDERLINE}This is underlined text{END}")
        print(f"{BOLD}{UNDERLINE}This is both bold and underlined{END}")

        for entity_text in list(entities_dict.keys()):
            pages_str = ", ".join(page for page in entities_dict[entity_text]["pages"])
            print(f"\n{GREEN}{entity_text}:{END} {YELLOW}(pages mentioned: {pages_str}){END}")
            for mention_index, mention in enumerate(entities_dict[entity_text]["mentions"]):
                mention_start_index = entities_dict[entity_text]["mention_starts"][mention_index]
                mention_end_index = entities_dict[entity_text]["mention_ends"][mention_index]
                mention_print_start_index = max(mention_start_index - 50, 0)
                mention_print_end_index = min(mention_end_index + 50, len(mention))
                page_mention_text = entities_dict[entity_text]['pages'][mention_index] + " - s:" + str(entities_dict[entity_text]['segment_numbers'][mention_index])
                mention_text = (f"{PINK}-{END}{ITALIC}[{page_mention_text}]{END} "
                                f"{PINK} ..." + mention[mention_print_start_index: mention_start_index] + f"{END}"
                                + f"{BOLD}{UNDERLINE}"+ mention[mention_start_index: mention_end_index] + f"{END}" +
                                f"{PINK}" + mention[mention_end_index: mention_print_end_index] + f"...{END}")

                print(mention_text)



def get_names():
    names_path = Path("./projects/pdf-entity-extraction/cejil_data/ner_text_files/person_names.txt")
    return names_path.read_text().split("\n")

def get_word_intersect_ratio(word1: str, word2: str):
    words1 = set(word1.lower().split())
    words2 = set(word2.lower().split())
    max_len = max(len(words1), len(words2))
    intersection = len(words1 & words2)
    return intersection / max_len


def find_unique_entities(entity_texts: list[str]):
    non_unique_entity_indexes = []
    for current_text_index, text_1 in enumerate(entity_texts):
        if current_text_index in non_unique_entity_indexes:
            continue
        for comparison_text_index, text_2 in enumerate(entity_texts):
            if current_text_index == comparison_text_index:
                continue
            if comparison_text_index in non_unique_entity_indexes:
                continue
            if ratio(text_1, text_2) > 0.79 or get_word_intersect_ratio(text_1, text_2) > 0.65:
                if current_text_index not in non_unique_entity_indexes:
                    non_unique_entity_indexes.append(current_text_index)
                non_unique_entity_indexes.append(comparison_text_index)
                continue
    non_unique_entities = [entity_texts[index] for index in non_unique_entity_indexes]
    unique_entities = [entity_texts[index] for index in range(len(entity_texts)) if index not in non_unique_entity_indexes]
    return sorted(non_unique_entities), sorted(unique_entities)


def get_ollama_extraction(entity_texts: list[str], content: str):
    # content = OllamaMultipleEntityExtractor.get_person_entity_prompt(entity_texts)
    client = Client(host=f"{HOST}")
    response = client.chat(model=f"{MODEL_NAME}", options={"temperature": 0},
                           messages=[{"role": "user", "content": content}])
    response_content = response["message"]["content"]
    extracted_entities_list = response_content.split("\n")
    return extracted_entities_list

def process_a_name_group(name_group, unique_names):
    non_unique_names_in_group, unique_names_in_group = find_unique_entities(name_group)
    unique_names.extend(unique_names_in_group)
    if not non_unique_names_in_group:
        return
    print(non_unique_names_in_group)
    content = OllamaMultipleEntityExtractor.get_person_entity_prompt(non_unique_names_in_group)
    ollama_extracted_entities = get_ollama_extraction(non_unique_names_in_group, content)
    unique_names.extend(ollama_extracted_entities)

def find_unique_names(names_list: list[str]):
    if not names_list:
        return

    name_group = [names_list[0]]
    unique_names = []

    for i in range(1, len(names_list)):
        current_name = names_list[i]

        if len(name_group) < 15:
            name_group.append(current_name)
            continue

        last_name_in_group = name_group[-1]
        is_similar = ratio(last_name_in_group, current_name) > 0.79 or get_word_intersect_ratio(current_name, last_name_in_group) > 0.65
        if is_similar:
            name_group.append(current_name)
            continue
        process_a_name_group(name_group, unique_names)
        name_group = [current_name]

    if name_group:
        process_a_name_group(name_group, unique_names)

    # unique_names.sort()
    output_path = Path(f"./projects/pdf-entity-extraction/cejil_data/ner_text_files/all_unique_person_names_{MODEL_NAME}.txt")
    output_path.write_text("\n".join(unique_names))


def is_name_similar(name_group: list[str], name_to_check: str):
    for name in name_group:
        if ratio(name, name_to_check) > 0.79 or get_word_intersect_ratio(name, name_to_check) > 0.65:
            return True
    return False


def process_a_name_group_with_similarity(name_group, unique_names, json_data):
    if len(name_group) < 2:
        unique_names.extend(name_group)
        return
    print(f"Processing group: \033[94m{name_group}\033[0m")
    content = OllamaMultipleEntityExtractor.get_person_entity_prompt(name_group)
    ollama_extracted_entities = get_ollama_extraction(name_group, content)
    print(f"Result: \033[92m{ollama_extracted_entities}\033[0m")
    processed_names = [
        name.strip()
        for entity in ollama_extracted_entities
        for name in entity.split(',') if entity
    ]

    unique_names.extend(processed_names)

    if not len(processed_names) == 1:
        return

    if not processed_names[0] in json_data.keys():
        return

    for name in name_group:
        if name not in json_data.keys():
            continue
        if name == processed_names[0]:
            continue
        json_data[processed_names[0]]["pages"].extend(json_data[name]["pages"])
        json_data[processed_names[0]]["mentions"].extend(json_data[name]["mentions"])
        json_data[processed_names[0]]["mention_starts"].extend(json_data[name]["mention_starts"])
        json_data[processed_names[0]]["mention_ends"].extend(json_data[name]["mention_ends"])
        json_data[processed_names[0]]["segment_numbers"].extend(json_data[name]["segment_numbers"])
        json_data.pop(name)



def get_similar_names_of_given_name(names_list: list[str], indexes_to_skip: list[int], name_to_check: str,
                                    current_index: int):
    similar_names = [name_to_check]
    for i in range(len(names_list)):
        if i == current_index or i in indexes_to_skip:
            continue
        if is_name_similar(similar_names, names_list[i]):
            similar_names.append(names_list[i])
            indexes_to_skip.append(i)
    return similar_names


def find_unique_names_from_similar_groups(names_list: list[str], sub_json_data):

    unique_names = []
    indexes_to_skip = []

    for i in range(len(names_list)):
        if i in indexes_to_skip:
            continue

        current_name = names_list[i]
        name_group = get_similar_names_of_given_name(names_list, indexes_to_skip, current_name, i)

        if name_group:
            process_a_name_group_with_similarity(name_group, unique_names, sub_json_data)
            indexes_to_skip.append(i)

    # unique_names.sort()
    output_path = Path(f"./projects/pdf-entity-extraction/cejil_data/ner_text_files/all_unique_person_names_{MODEL_NAME}.txt")
    output_path.write_text("\n".join(unique_names))

    output_path = Path(f"./projects/pdf-entity-extraction/cejil_data/ner_json_files/sub_person_names_{MODEL_NAME}.json")
    with output_path.open('w', encoding='utf-8') as f:
        json.dump(sub_json_data, f, ensure_ascii=False, indent=4)


def rebuild_entities_dict(original_entities_dict, name_to_unique_name_mapping):
    new_entities_dict = {'PERSON': {}}

    for original_name, data in original_entities_dict['PERSON'].items():
        unique_name = name_to_unique_name_mapping.get(original_name, original_name)
        if unique_name not in new_entities_dict['PERSON']:
            new_entities_dict['PERSON'][unique_name] = {
                "pages": set(),
                "mentions": set(),
                "pdf_names": []
            }

        new_entities_dict['PERSON'][unique_name]["pages"].update(data["pages"])
        new_entities_dict['PERSON'][unique_name]["mentions"].update(data["mentions"])
        new_entities_dict['PERSON'][unique_name]["pdf_names"].extend(data["pdf_names"])

    # Convert sets to lists
    for person_data in new_entities_dict['PERSON'].values():
        person_data["pages"] = list(person_data["pages"])
        person_data["mentions"] = list(person_data["mentions"])

    return new_entities_dict


if __name__ == '__main__':
    # extract_and_save_entities()
    # save_to_text_file()



    # json_data = json.loads(Path("./projects/pdf-entity-extraction/cejil_data/ner_json_files/person_names.json").read_text())
    # # print(json_data.keys())
    #
    #
    # # names_list: list[str] = []
    # # for pdf_name in json_data:
    # #     for name_entity in list(json_data[pdf_name]["PERSON"].keys()):
    # #         names_list.append(name_entity)
    #
    #
    #
    #
    # names_list = get_names()[:210]
    # sub_json_data = {k: json_data[k] for k in names_list}
    #
    # # find_unique_names(names_list)
    # start = time()
    # find_unique_names_from_similar_groups(names_list, sub_json_data)
    # print("Unique name extraction finished in", round(time() - start, 2), "seconds")


    sub_person_names = json.loads(Path(f"./projects/pdf-entity-extraction/cejil_data/ner_json_files/sub_person_names_{MODEL_NAME}.json").read_text())
    OllamaMultipleEntityExtractor.print_formatted_entities(sub_person_names)


# All Unique name extraction finished in 247.15 seconds
