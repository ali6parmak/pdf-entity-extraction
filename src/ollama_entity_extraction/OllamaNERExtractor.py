import json
from pathlib import Path

from Levenshtein import ratio
from ollama import Client
from spacy.language import Language
from flair.nn import Classifier

from ollama_entity_extraction.data_model.ConsoleTextColor import ConsoleTextColor
from ollama_entity_extraction.data_model.ConsoleTextStyle import ConsoleTextStyle
from ollama_entity_extraction.data_model.EntitiesDict import EntitiesDict
from ollama_entity_extraction.data_model.EntityInfo import EntityInfo

OLLAMA_HOST = "http://localhost:11434"


class OllamaNERExtractor:
    def __init__(
        self, ner_model: Classifier | Language = None, ollama_model_name: str = "llama3.1", ollama_host: str = OLLAMA_HOST
    ):
        if ner_model is None:
            ner_model = Classifier.load("ner-ontonotes-large")
        self.ner_model = ner_model
        self.ollama_client = Client(host=ollama_host)
        self.ollama_model_name = ollama_model_name

    @staticmethod
    def save_entity_texts(save_path: str | Path, entity_texts: list[str]):
        output_path = Path(save_path)
        entity_texts.sort()
        output_path.write_text("\n".join(entity_texts))

    @staticmethod
    def save_entities_dict(save_path: str | Path, entities_dict: EntitiesDict):
        output_path = Path(save_path)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(entities_dict.to_dict(), f, ensure_ascii=False, indent=4)

    @staticmethod
    def get_prompt(entity_texts: list[str]) -> str:
        raise NotImplementedError("This method should be implemented in the subclass.")

    @staticmethod
    def get_word_intersection_ratio(word1: str, word2: str) -> float:
        words1 = set(word1.lower().split())
        words2 = set(word2.lower().split())
        max_len = max(len(words1), len(words2))
        if max_len == 0:
            return 0.0
        intersection = len(words1 & words2)
        return intersection / max_len

    @staticmethod
    def _format_mention(
        entities_dict: EntitiesDict,
        entity_text: str,
        mention_index: int,
    ) -> str:
        context_size = 50

        entity_info = entities_dict.get_entity_info(entity_text)
        if entity_info is None:
            return ""

        mention = entity_info.mentions[mention_index]
        page_info = entity_info.pages[mention_index]
        segment_number = entity_info.segment_numbers[mention_index]
        mentioned_segments_text = f"{page_info} - s:{segment_number}"

        mention_start_index = entity_info.mention_starts[mention_index]
        mention_end_index = entity_info.mention_ends[mention_index]

        mention_print_start_index = max(mention_start_index - context_size, 0)
        mention_print_end_index = min(mention_end_index + context_size, len(mention))

        mentioned_segments_text = ConsoleTextColor.PINK("-") + ConsoleTextStyle.ITALIC(f"[{mentioned_segments_text}]")
        mention_content = ConsoleTextColor.PINK(" ... " + mention[mention_print_start_index:mention_start_index])

        entity_text_styles = [ConsoleTextStyle.BOLD, ConsoleTextStyle.UNDERLINE]
        entity_text_formatted = ConsoleTextStyle.apply(entity_text, entity_text_styles)

        mention_end_text = ConsoleTextColor.PINK(mention[mention_end_index:mention_print_end_index] + "...")

        formatted_mention = mentioned_segments_text + mention_content + entity_text_formatted + mention_end_text

        return formatted_mention

    @staticmethod
    def print_formatted_entities(entities_dict: EntitiesDict):
        for entity_text, entity_info in entities_dict.items():
            pages_str = ", ".join(entity_info.pages)
            print(
                f"\n{ConsoleTextColor.GREEN(entity_text)}: " f"{ConsoleTextColor.YELLOW(f'(pages mentioned: {pages_str})')}"
            )
            for mention_index in range(len(entity_info.mentions)):
                formatted_mention = OllamaNERExtractor._format_mention(entities_dict, entity_text, mention_index)
                print(formatted_mention)

    @staticmethod
    def remove_overlapping_entities(entities, start_key="start_index", end_key="end_index") -> list[dict[str, str | int]]:
        sorted_entities = sorted(entities, key=lambda x: (x[start_key], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity[start_key] >= last_end:
                result.append(entity)
                last_end = entity[end_key]

        return result

    def get_ollama_extraction(self, content: str, options=None) -> list[str]:
        response = self.ollama_client.chat(
            model=self.ollama_model_name, options=options, messages=[{"role": "user", "content": content}]
        )
        response_content = response["message"]["content"]
        extracted_entities_list = response_content.split("\n")
        return extracted_entities_list

    def are_entities_similar(self, entity_group: list[str], entity_to_check: str) -> bool:
        for entity in entity_group:
            if ratio(entity, entity_to_check) > 0.79 or self.get_word_intersection_ratio(entity, entity_to_check) > 0.65:
                return True
        return False

    def process_entity_group(self, entity_group: list[str], entities_dict: EntitiesDict):
        if len(entity_group) < 2:
            return
        print(f"Processing group: \033[94m{entity_group}\033[0m")
        content = self.get_prompt(entity_group)
        ollama_extracted_entities = self.get_ollama_extraction(content, options={"temperature": 0})
        normalized_entities = list(set([
            # sub_entity.strip() for entity in ollama_extracted_entities for sub_entity in entity.split(",") if entity
            entity.strip() for entity in ollama_extracted_entities
        ]))
        print(f"Result: \033[92m{normalized_entities}\033[0m")

        if not len(normalized_entities) == 1:
            return

        representative_entity = normalized_entities[0]
        if representative_entity not in entities_dict.keys():
            entities_dict.entities[representative_entity] = EntityInfo()

        for entity in entity_group:
            if entity != representative_entity:
                entities_dict.merge_entities(representative_entity, entity)

    def get_similar_entities_of_given_entity(
        self, entities_list: list[str], indexes_to_skip: list[int], entity_to_check: str, current_index: int
    ) -> list[str]:
        similar_entities: list[str] = [entity_to_check]
        for i in range(len(entities_list)):
            if i == current_index or i in indexes_to_skip:
                continue
            if self.are_entities_similar(similar_entities, entities_list[i]):
                similar_entities.append(entities_list[i])
                indexes_to_skip.append(i)
        return similar_entities

    def find_unique_entities(self, entities_dict: EntitiesDict) -> EntitiesDict:

        entities_dict.sort_entities()
        indexes_to_skip = []
        entities_list: list[str] = list(entities_dict.keys())

        for i, current_entity in enumerate(entities_list):
            if i in indexes_to_skip:
                continue

            entity_group = self.get_similar_entities_of_given_entity(entities_list, indexes_to_skip, current_entity, i)

            if entity_group:
                self.process_entity_group(entity_group, entities_dict)
                indexes_to_skip.append(i)

        return entities_dict

    def extract_entities_from_text(self, text: str) -> list[dict[str, str | int]]:
        pass

    def extract_entities(self, pdf_name: str, segment_boxes: list[dict], entities_dict: EntitiesDict = None) -> EntitiesDict:
        if entities_dict is None:
            entities_dict = EntitiesDict()
        current_page = 1
        segment_no = 1
        for segment_box in segment_boxes:
            page_number = segment_box["page_number"]
            if page_number != current_page:
                current_page = page_number
                segment_no = 1
            reconstructed_text = " ".join([word for word in segment_box["text"].split()])
            if not reconstructed_text:
                continue
            segment_entities = self.extract_entities_from_text(reconstructed_text)
            for entity in segment_entities:
                entity_text = entity["text"]
                entity_start_index = entity["start_index"]
                entity_end_index = entity["end_index"]

                page_info = f"{pdf_name} - p:{page_number}"
                entities_dict.add_entity(
                    entity_text=entity_text,
                    page_info=page_info,
                    mention=reconstructed_text,
                    mention_start=entity_start_index,
                    mention_end=entity_end_index,
                    segment_number=segment_no,
                )
            segment_no += 1

        return entities_dict
