import json
from pathlib import Path
from typing import List, Dict, Union, Optional, Tuple

from ollama import Client
from spacy.language import Language
from flair.data import Sentence
from flair.models import SequenceTagger  # Updated import for the NER model

from ollama_entity_extraction.data_model.ConsoleTextColor import ConsoleTextColor
from ollama_entity_extraction.data_model.ConsoleTextStyle import ConsoleTextStyle

from difflib import SequenceMatcher  # For string similarity calculations
from dataclasses import dataclass, field

OLLAMA_HOST = "http://localhost:11434"


@dataclass
class EntityInfo:
    pages: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    mention_starts: List[int] = field(default_factory=list)
    mention_ends: List[int] = field(default_factory=list)
    segment_numbers: List[int] = field(default_factory=list)

    def extend(self, other: "EntityInfo"):
        self.pages.extend(other.pages)
        self.mentions.extend(other.mentions)
        self.mention_starts.extend(other.mention_starts)
        self.mention_ends.extend(other.mention_ends)
        self.segment_numbers.extend(other.segment_numbers)


class EntitiesDict:
    def __init__(self):
        self.entities: Dict[str, EntityInfo] = {}

    def add_entity(
        self,
        entity_text: str,
        page_info: str,
        mention: str,
        mention_start: int,
        mention_end: int,
        segment_number: int,
    ):
        if entity_text not in self.entities:
            self.entities[entity_text] = EntityInfo()
        entity_info = self.entities[entity_text]
        entity_info.pages.append(page_info)
        entity_info.mentions.append(mention)
        entity_info.mention_starts.append(mention_start)
        entity_info.mention_ends.append(mention_end)
        entity_info.segment_numbers.append(segment_number)

    def merge_entities(self, target_entity: str, source_entity: str):
        if target_entity not in self.entities:
            self.entities[target_entity] = EntityInfo()
        if source_entity in self.entities:
            self.entities[target_entity].extend(self.entities[source_entity])
            del self.entities[source_entity]

    def to_dict(self) -> Dict[str, Dict]:
        return {entity_text: entity_info.__dict__ for entity_text, entity_info in self.entities.items()}

    @staticmethod
    def from_dict(data: Dict[str, Dict]) -> "EntitiesDict":
        entities_dict = EntitiesDict()
        for entity_text, entity_info_data in data.items():
            entity_info = EntityInfo(**entity_info_data)
            entities_dict.entities[entity_text] = entity_info
        return entities_dict

    def keys(self):
        return self.entities.keys()

    def items(self):
        return self.entities.items()

    def get_entity_info(self, entity_text: str) -> Optional[EntityInfo]:
        return self.entities.get(entity_text)

    def pop(self, entity_text: str):
        return self.entities.pop(entity_text, None)


class OllamaNERExtractor:
    def __init__(
        self,
        ner_model: Optional[Union[SequenceTagger, Language]] = None,
        ollama_model_name: str = "llama3.1",
        ollama_host: str = OLLAMA_HOST,
    ):
        if ner_model is None:
            ner_model = SequenceTagger.load("ner-ontonotes-large")
        self.ner_model = ner_model
        self.ollama_client = Client(host=ollama_host)
        self.ollama_model_name = ollama_model_name

    @staticmethod
    def get_prompt(entity_texts: List[str]) -> str:
        # Placeholder method to be overridden by subclasses
        raise NotImplementedError("Subclasses should implement this method.")

    @staticmethod
    def save_entity_texts(save_path: Union[str, Path], entity_texts: List[str]):
        output_path = Path(save_path)
        output_path.write_text("\n".join(entity_texts), encoding="utf-8")

    @staticmethod
    def save_entities_dict(save_path: Union[str, Path], entities_dict: EntitiesDict):
        output_path = Path(save_path)
        data = entities_dict.to_dict()
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    def get_ollama_extraction(self, content: str, options: Optional[Dict] = None):
        if options is None:
            options = {}
        response = self.ollama_client.chat(
            model=self.ollama_model_name, options=options, messages=[{"role": "user", "content": content}]
        )
        response_content = response["message"]["content"]
        extracted_entities_list = response_content.strip().split("\n")
        return extracted_entities_list

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
    def remove_overlapping_entities(
        entities: List[Dict[str, Union[str, int]]], start_key: str = "start_index", end_key: str = "end_index"
    ) -> List[Dict[str, Union[str, int]]]:
        sorted_entities = sorted(entities, key=lambda x: (x[start_key], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity[start_key] >= last_end:
                result.append(entity)
                last_end = entity[end_key]

        return result

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Union[str, int]]]:
        # Placeholder method to be overridden by subclasses
        raise NotImplementedError("Subclasses should implement this method.")

    def extract_entities(
        self, pdf_name: str, segment_boxes: List[Dict], entities_dict: Optional[EntitiesDict] = None
    ) -> EntitiesDict:
        if entities_dict is None:
            entities_dict = EntitiesDict()
        current_page = 1
        segment_no = 1
        for segment_box in segment_boxes:
            page_number = segment_box.get("page_number", current_page)
            if page_number != current_page:
                current_page = page_number
                segment_no = 1
            reconstructed_text = " ".join(segment_box.get("text", "").split())
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


class OllamaNameExtractor(OllamaNERExtractor):
    @staticmethod
    def get_prompt(entity_texts: List[str]) -> str:
        entities_string = "\n".join(entity_texts)
        content = (
            "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of person names "
            "and consolidate entries that refer to the same individual, even if there are minor variations due to typos, accents, or missing names."
            "\n\n"
            "Here are the instructions that you should follow:\n"
            "\n - Identify names that refer to the same person."
            '\n - Consider variations such as spelling differences, missing middle names, or presence of accents (e.g., "Ali" vs. "Alí").'
            "\n - For each group of similar names, choose the most complete and accurate version as the representative name."
            "\n - Prefer names that include full middle names over those that have initials or omit them."
            "\n - Do not skip any entities in the output, return all the unique entities."
            "\n - Do not create new, additional entities, just output the consolidated list."
            "\n - Do not make any additional explanations, just output the entities."
            "\n\n\nHere are the names to process:\n\n"
            "### INPUT\n\n"
            f"{entities_string}\n\n\n"
            f"### OUTPUT:"
        )
        return content

    @staticmethod
    def get_word_intersect_ratio(word1: str, word2: str) -> float:
        words1 = set(word1.lower().split())
        words2 = set(word2.lower().split())
        max_len = max(len(words1), len(words2))
        if max_len == 0:
            return 0.0
        intersection = len(words1 & words2)
        return intersection / max_len

    def extract_entities_from_text(self, text: str) -> List[Dict[str, Union[str, int]]]:
        entity_results = []

        sentence = Sentence(text)
        self.ner_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag == "PERSON":
                entity_results.append(
                    {
                        "text": entity.text.title(),
                        "entity_label": entity.tag,
                        "start_index": entity.start_position,
                        "end_index": entity.end_position,
                    }
                )

        entities = self.remove_overlapping_entities(entity_results)
        return entities

    def is_name_similar(self, name_group: List[str], name_to_check: str) -> bool:
        for name in name_group:
            similarity_ratio = SequenceMatcher(None, name, name_to_check).ratio()
            word_intersect_ratio = self.get_word_intersect_ratio(name, name_to_check)
            if similarity_ratio > 0.79 or word_intersect_ratio > 0.65:
                return True
        return False

    def process_name_group(self, name_group: List[str], unique_names: List[str], entities_dict: EntitiesDict):
        if len(name_group) < 2:
            unique_names.extend(name_group)
            return
        print(f"Processing group: \033[94m{name_group}\033[0m")
        content = self.get_prompt(name_group)
        ollama_extracted_entities = self.get_ollama_extraction(content, options={"temperature": 0})
        print(f"Result: \033[92m{ollama_extracted_entities}\033[0m")
        processed_names = [
            name.strip() for entity in ollama_extracted_entities for name in entity.split(",") if entity.strip()
        ]

        unique_names.extend(processed_names)

        if len(processed_names) != 1:
            return

        representative_name = processed_names[0]
        if representative_name not in entities_dict.keys():
            entities_dict.entities[representative_name] = EntityInfo()

        for name in name_group:
            if name != representative_name:
                entities_dict.merge_entities(representative_name, name)

    def get_similar_names_of_given_name(
        self, names_list: List[str], indexes_to_skip: List[int], name_to_check: str, current_index: int
    ) -> List[str]:
        similar_names = [name_to_check]
        for i, other_name in enumerate(names_list):
            if i == current_index or i in indexes_to_skip:
                continue
            if self.is_name_similar(similar_names, other_name):
                similar_names.append(other_name)
                indexes_to_skip.append(i)
        return similar_names

    def find_unique_names_from_similar_groups(
        self, names_list: List[str], entities_dict: EntitiesDict
    ) -> Tuple[List[str], EntitiesDict]:
        unique_names = []
        indexes_to_skip = []

        for i, current_name in enumerate(names_list):
            if i in indexes_to_skip:
                continue

            name_group = self.get_similar_names_of_given_name(names_list, indexes_to_skip, current_name, i)

            if name_group:
                self.process_name_group(name_group, unique_names, entities_dict)
                indexes_to_skip.extend([j for j, name in enumerate(names_list) if name in name_group and j != i])

        return unique_names, entities_dict
