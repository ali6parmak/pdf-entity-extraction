from ollama import Client
from spacy.language import Language
from flair.nn import Classifier



class OllamaNERExtractor:
    def __init__(self, ner_model: Classifier | Language, ollama_host: str = "http://localhost:11434"):
        self.ner_model = ner_model
        self.ollama_client = Client(host=ollama_host)

    @staticmethod
    def get_prompt() -> str:
        pass

    def get_ollama_extraction(self, model_name: str, content: str, options=None):
        response = self.ollama_client.chat(model=model_name, options=options,
                                           messages=[{"role": "user", "content": content}])
        response_content = response["message"]["content"]
        extracted_entities_list = response_content.split("\n")
        return extracted_entities_list

    @staticmethod
    def print_formatted_entities(entities_dict):
        PINK = '\033[95m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        BOLD = '\033[1m'
        ITALIC = '\033[3m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

        for entity_text in list(entities_dict.keys()):
            pages_str = ", ".join(page for page in entities_dict[entity_text]["pages"])
            print(f"\n{GREEN}{entity_text}:{END} {YELLOW}(pages mentioned: {pages_str}){END}")
            for mention_index, mention in enumerate(entities_dict[entity_text]["mentions"]):
                mention_start_index = entities_dict[entity_text]["mention_starts"][mention_index]
                mention_end_index = entities_dict[entity_text]["mention_ends"][mention_index]
                mention_print_start_index = max(mention_start_index - 50, 0)
                mention_print_end_index = min(mention_end_index + 50, len(mention))

                page_mention_text = (entities_dict[entity_text]['pages'][mention_index] +
                                     " - s:" + str(entities_dict[entity_text]['segment_numbers'][mention_index]))

                mention_text = (f"{PINK}-{END}{ITALIC}[{page_mention_text}]{END} "
                                f"{PINK} ..." + mention[mention_print_start_index: mention_start_index] + f"{END}"
                                + f"{BOLD}{UNDERLINE}"+ mention[mention_start_index: mention_end_index] + f"{END}" +
                                f"{PINK}" + mention[mention_end_index: mention_print_end_index] + f"...{END}")

                print(mention_text)

    @staticmethod
    def remove_overlapping_entities(entities, start_key="start_index", end_key="end_index") -> list:
        sorted_entities = sorted(entities, key=lambda x: (x[start_key], -len(x["text"])))

        result = []
        last_end = -1

        for entity in sorted_entities:
            if entity[start_key] >= last_end:
                result.append(entity)
                last_end = entity[end_key]

        return result

    @staticmethod
    def create_default_dict() -> dict[str, list]:
        return {"pages": list(), "mentions": list(), "mention_starts": list(), "mention_ends": list(),
                "segment_numbers": list()}

    def extract_entities_from_text(self, text: str) -> list[list[str, str]]:
        pass

    def extract_entities(self, pdf_name: str, segment_boxes: list[dict], entities_dict=None):
        if not entities_dict:
            entities_dict = {}
        current_page = 1
        segment_no = 1
        for segment_box in segment_boxes:
            if segment_box["page_number"] != current_page:
                current_page = segment_box["page_number"]
                segment_no = 1
            reconstructed_text = " ".join([word for word in segment_box["text"].split()])
            if not reconstructed_text:
                continue
            segment_entities = self.extract_entities_from_text(reconstructed_text)
            for entity_text, entity_label in segment_entities:
                entity_text_title = entity_text.title()
                if entity_text_title not in entities_dict:
                    entities_dict[entity_text_title] = self.create_default_dict()

                mention_start = reconstructed_text.index(entity_text)
                mention_end = reconstructed_text.index(entity_text) + len(entity_text)
                entities_dict[entity_text_title]["pages"].append(pdf_name + " - p:" + str(segment_box["page_number"]))
                entities_dict[entity_text_title]["mentions"].append(reconstructed_text)
                entities_dict[entity_text_title]["mention_starts"].append(mention_start)
                entities_dict[entity_text_title]["mention_ends"].append(mention_end)
                entities_dict[entity_text_title]["segment_numbers"].append(segment_no)
            segment_no += 1


        return entities_dict

