from Levenshtein import ratio
from flair.data import Sentence
from ollama_entity_extraction.OllamaNERExtractor import OllamaNERExtractor
from ollama_entity_extraction.data_model.EntitiesDict import EntitiesDict
from ollama_entity_extraction.data_model.EntityInfo import EntityInfo


class OllamaLocationExtractor(OllamaNERExtractor):
    @staticmethod
    def get_prompt(entity_texts: list[str]) -> str:
        entities_string = "\n".join(entity_texts)
        content = (
            "You are an expert in data cleaning and entity resolution, specializing in geographical and geopolitical data. "
            "Your task is to analyze a list of geopolitical entity names (GPE entities) and consolidate entries that refer to the same location, "
            "even if there are variations in spelling, abbreviations, or naming conventions."
            "\n\n"
            "Here are the instructions that you should follow:\n"
            "\n - Identify names that refer to the same location or place."
            "\n - Consider variations such as spelling differences, abbreviations, alternate names, former names, or common misspellings. Do not talk about the corrections."
            "\n - For each group of similar or equivalent names, choose the most widely recognized and official name as the representative. Do not talk about the corrections."
            # "\n - Normalize country names, state/province names, cities, and other geopolitical entities to their standard forms. Do not talk about the corrections."
            # "\n - Use internationally recognized names (e.g., 'Myanmar' instead of 'Burma', 'Eswatini' instead of 'Swaziland')."
            "\n - Be aware of alternative spellings or names in different languages (e.g., 'South Korea' and 'Republic of Korea'). Do not talk about the corrections."
            "\n - Do not skip any entities in the output, return all the unique entities."
            "\n - Do not create new, additional entities, just output the consolidated list."
            "\n - Do not include any additional explanations; just output the consolidated list of unique GPE entities."
            "\n - Do not make any comment, do not put any note, do not say \"Here are the entities\", ONLY output the consolidated list."
            "\n - Only work on the following GPE entities in the input."
            "\n\n\nHere are the entities to process:\n\n"
            "### INPUT\n\n"
            f"{entities_string}\n\n"
            "### OUTPUT:"
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

    def extract_entities_from_text(self, text: str) -> list[dict[str, str | int]]:
        entity_results = []

        sentence = Sentence(text)
        self.ner_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag != "GPE":
                continue
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

    def is_name_similar(self, name_group: list[str], name_to_check: str) -> bool:
        for name in name_group:
            if ratio(name, name_to_check) > 0.79 or self.get_word_intersect_ratio(name, name_to_check) > 0.65:
                return True
        return False

    def process_name_group(self, name_group: list[str], unique_names: list[str], entities_dict: EntitiesDict):
        if len(name_group) < 2:
            unique_names.extend(name_group)
            return
        print(f"Processing group: \033[94m{name_group}\033[0m")
        content = self.get_prompt(name_group)
        ollama_extracted_entities = self.get_ollama_extraction(content, options={"temperature": 0})
        print(f"Result: \033[92m{ollama_extracted_entities}\033[0m")
        processed_names = [name.strip() for entity in ollama_extracted_entities for name in entity.split(",") if entity]

        unique_names.extend(processed_names)

        if not len(processed_names) == 1:
            return

        representative_name = processed_names[0]
        if representative_name not in entities_dict.keys():
            entities_dict.entities[representative_name] = EntityInfo()

        for name in name_group:
            if name != representative_name:
                entities_dict.merge_entities(representative_name, name)

    def get_similar_names_of_given_name(
        self, names_list: list[str], indexes_to_skip: list[int], name_to_check: str, current_index: int
    ):
        similar_names = [name_to_check]
        for i in range(len(names_list)):
            if i == current_index or i in indexes_to_skip:
                continue
            if self.is_name_similar(similar_names, names_list[i]):
                similar_names.append(names_list[i])
                indexes_to_skip.append(i)
        return similar_names

    def find_unique_names(self, entities_dict: EntitiesDict) -> tuple[list[str], EntitiesDict]:

        entities_dict.sort_entities()
        unique_names = []
        indexes_to_skip = []
        names_list: list[str] = list(entities_dict.keys())

        for i, current_name in enumerate(names_list):
            if i in indexes_to_skip:
                continue

            name_group = self.get_similar_names_of_given_name(names_list, indexes_to_skip, current_name, i)

            if name_group:
                self.process_name_group(name_group, unique_names, entities_dict)
                indexes_to_skip.append(i)

        return unique_names, entities_dict
