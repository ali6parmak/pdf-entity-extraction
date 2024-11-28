from flair.data import Sentence
from ollama_entity_extraction.OllamaNERExtractor import OllamaNERExtractor


class OllamaOrganizationExtractor(OllamaNERExtractor):
    @staticmethod
    def get_prompt(entity_texts: list[str]) -> str:
        entities_string = "\n".join(entity_texts)
        content = (
            "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of organization names "
            "and consolidate entries that refer to the same organization, even if there are minor variations due to typos, abbreviations, or naming differences."
            "\n\n"
            "Here are the instructions that you should follow:\n"
            "\n - Identify names that refer to the same organization."
            '\n - Consider variations such as spelling differences, abbreviations, acronyms, or presence of special characters (e.g., "IBM" vs. "I.B.M.").'
            "\n - For each group of similar names, choose the most complete and accurate version as the representative name."
            "\n - Prefer official full names over abbreviations or colloquial versions."
            "\n - Do not skip any entities in the output, return all the unique organization names."
            "\n - Do not make any additional explanations, comments, or clarifications, just output the entities."
            "\n\n\nHere are the names to process:\n\n"
            "### INPUT\n\n"
            f"{entities_string}\n\n\n"
            f"### OUTPUT:"
        )
        return content

    def extract_entities_from_text(self, text: str) -> list[dict[str, str | int]]:
        entity_results = []

        sentence = Sentence(text)
        self.ner_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag != "ORG":
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
