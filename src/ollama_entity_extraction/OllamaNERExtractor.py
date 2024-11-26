import json
from pathlib import Path

from ollama import Client
from spacy.language import Language
from flair.nn import Classifier

from ollama_entity_extraction.data_model.ConsoleTextColor import ConsoleTextColor
from ollama_entity_extraction.data_model.ConsoleTextStyle import ConsoleTextStyle
from ollama_entity_extraction.data_model.EntitiesDict import EntitiesDict

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
    def get_prompt(entity_texts: list[str]) -> str:
        raise NotImplementedError("This method should be implemented in the subclass.")

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

    def get_ollama_extraction(self, content: str, options=None):
        response = self.ollama_client.chat(
            model=self.ollama_model_name, options=options, messages=[{"role": "user", "content": content}]
        )
        response_content = response["message"]["content"]
        extracted_entities_list = response_content.split("\n")
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
    def remove_overlapping_entities(entities, start_key="start_index", end_key="end_index") -> list[dict[str, str | int]]:
        sorted_entities = sorted(entities, key=lambda x: (x[start_key], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity[start_key] >= last_end:
                result.append(entity)
                last_end = entity[end_key]

        return result

    def extract_entities_from_text(self, text: str) -> list[dict[str, str | int]]:
        pass

    def extract_entities(self, pdf_name: str, segment_boxes: list[dict], entities_dict: EntitiesDict = None):
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
