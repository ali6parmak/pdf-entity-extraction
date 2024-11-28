from flair.data import Sentence
from ollama_entity_extraction.OllamaNERExtractor import OllamaNERExtractor


class OllamaNameExtractor(OllamaNERExtractor):
    @staticmethod
    def get_prompt(entity_texts: list[str]) -> str:
        entities_string = "\n".join(entity_texts)
        content = (
            "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of person names "
            "and consolidate entries that refer to the same individual, even if there are minor variations due to typos, accents, or missing names."
            "\n\n"
            "Here are the instructions that you should follow:\n"
            "\n - Identify names that refer to the same person."
            '\n - Consider variations such as spelling differences, missing middle names, or presence of accents (e.g., "Ali" vs. "Alí"). Do not talk about corrections.'
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

    def extract_entities_from_text(self, text: str) -> list[dict[str, str | int]]:
        entity_results = []

        sentence = Sentence(text)
        self.ner_model.predict(sentence)
        flair_result = sentence.get_spans("ner")

        for entity in flair_result:
            if entity.tag not in {"PERSON"}:
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
