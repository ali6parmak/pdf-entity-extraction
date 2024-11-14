import json
from collections import defaultdict
from time import time

import spacy
from dateparser.search import search_dates
from flair.data import Sentence
from flair.nn import Classifier
from gliner import GLiNER
from configuration import CACHED_DATA_PATH


class MultipleEntityExtractor:
    def __init__(self):
        self.gliner_model = GLiNER.from_pretrained("urchade/gliner_multi-v2.1")
        self.opennyai_model = spacy.load("en_legal_ner_sm")
        self.flair_model = Classifier.load("ner-ontonotes-large")

    @staticmethod
    def find_unique_entity_dicts(entities: list[dict]) -> list[dict]:
        dicts_without_score = [{k: v for k, v in d.items() if k != "score"} for d in entities]
        return list({json.dumps(d, sort_keys=True): d for d in dicts_without_score}.values())

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

    def extract_date_entities(self, words: list[str]):

        entities = []
        window_size = 20
        slide_size = 10
        last_slide_end_index = 0

        for i in range(0, len(words), slide_size):
            window_words = words[i : i + window_size]
            window_text = " ".join(window_words)
            window_entities = self.gliner_model.predict_entities(window_text, ["date"])

            for entity in window_entities:
                entity["start"] += last_slide_end_index
                entity["end"] += last_slide_end_index

            slide_words = words[i : i + slide_size]
            slide_text = " ".join(slide_words)
            last_slide_end_index += len(slide_text) + 1
            entities.extend(window_entities)

        entities = self.find_unique_entity_dicts(entities)
        entities = [e for e in entities if search_dates(e["text"])]
        entities = self.remove_overlapping_entities(entities, "start", "end")
        date_times = [d[1] for e in entities for d in search_dates(e["text"])]
        return date_times


    def extract_entities_from_text(self, text: str):
        words = text.split()
        date_times = self.extract_date_entities(words)
        date_strings = [d.date() for d in date_times]

        entities = []

        for date_string in date_strings:
            entities.append([date_string, "DATE"])


        opennyai_result = self.opennyai_model(text)
        entity_results = []
        for entity in opennyai_result.ents:
            if entity.label_ not in {"PROVISION", "STATUTE", "CASE_NUMBER", "COURT", "PRECEDENT"}:
                continue

            entity_results.append(
                {
                    "text": entity.text,
                    "entity_label": entity.label_,
                    "start_index": entity.start_char,
                    "end_index": entity.end_char,
                }
            )

        sentence = Sentence(text)
        self.flair_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag not in {"ORG", "PERSON", "LAW", "GPE"}:
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

    def extract_entities(self, pdf_name: str):
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{pdf_name}.json").read_text())
        # entities_by_page = {}
        # entities_dict = defaultdict(lambda: defaultdict(set))
        # entities_dict = defaultdict(lambda: defaultdict(lambda: {"pages": set(), "mentions": set(), "mention_starts": []}))
        entities_dict = defaultdict(lambda: defaultdict(lambda: {"pages": set(), "mentions": set()}))

        for segment_box in segment_boxes:
            reconstructed_text = " ".join([word for word in segment_box["text"].split()])
            segment_entities = self.extract_entities_from_text(reconstructed_text)
            # entities_by_page.setdefault(segment_box["page_number"], []).extend(segment_entities)
            for entity_text, entity_label in segment_entities:
                entities_dict[entity_label][entity_text]["pages"].add(segment_box["page_number"])
                entities_dict[entity_label][entity_text]["mentions"].add(reconstructed_text)
                # entities_dict[entity_label][entity_text]["mention_starts"].append(reconstructed_text.index(entity_text))

        # for page_number, entities in entities_by_page.items():
        #     for entity_text, entity_label in entities:
        #         entities_dict[entity_label][entity_text].add(page_number)

        return entities_dict

    @staticmethod
    def print_formatted_entities(entities_dict):
        entity_types = ["ORG", "PERSON", "LAW", "DATE", "GPE", "PROVISION", "STATUTE", "CASE_NUMBER", "COURT"]

        for entity_type in entity_types:
            if entity_type in entities_dict:
                print(f"\n\033[94m{entity_type}:\033[0m")
                sorted_entities = sorted(entities_dict[entity_type].items())

                for entity_text, data in sorted_entities:
                    pages_str = ", ".join(str(page) for page in sorted(data["pages"]))
                    print(f"\033[92m{entity_text}\033[0m")

                    # for mention_index, mention in enumerate(sorted(data["mentions"])):
                    #     # mention_start = min(data['mention_starts'][mention_index], 0)
                    #     # mention_end = min(len(mention), data['mention_starts'] + 50)
                    #     # print(f"\033[95m- ... {mention[mention_start: mention_end]} ...\033[0m")
                    #     print(f"\033[95m- {mention}\033[0m")


if __name__ == '__main__':
    extractor = MultipleEntityExtractor()
    start = time()
    entities_dict = extractor.extract_entities("cejil_staging33")
    print("Extraction finished in", round(time() - start, 2), "seconds")
    extractor.print_formatted_entities(entities_dict)
    # Extraction finished in 120.88 seconds
    # Extraction finished in 117 seconds (mentions added)
