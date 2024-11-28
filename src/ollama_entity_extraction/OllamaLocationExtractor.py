from flair.data import Sentence
from ollama_entity_extraction.OllamaNERExtractor import OllamaNERExtractor


class OllamaLocationExtractor(OllamaNERExtractor):
    @staticmethod
    def get_prompt(entity_texts: list[str]) -> str:
        entities_string = "\n".join(entity_texts)
        content = (
            "You are an expert in data cleaning and entity resolution, specializing in geographical and geopolitical data. "
            "Your task is to analyze a list of geographical names (GPE entities) and consolidate entries that refer to the same specific location, "
            "even if there are variations in spelling, abbreviations, or naming conventions."
            "\n\n"
            "Here are the instructions you should follow:\n"
            "\n - Identify names that refer to the **same specific location or place at the same geographical level** (e.g., city with city, country with country)."
            "\n - **Do not generalize**: Do not replace a specific location with a broader one (e.g., do not replace 'Bogotá' with 'Colombia')."
            "\n - **Do not introduce new entities**: Only use the entities provided in the input; do not add additional locations or countries."
            # "\n - Normalize country names, state/province names, cities, and other geopolitical entities to their standard forms. Do not talk about the corrections."
            # "\n - Use internationally recognized names (e.g., 'Myanmar' instead of 'Burma', 'Eswatini' instead of 'Swaziland')."
            "\n - Consider variations such as spelling differences, abbreviations, alternate names, former names, or common misspellings."
            "\n - For each group of similar or equivalent names, choose the most widely recognized and official name as the representative."
            "\n - Be aware of alternative spellings or names in different languages (e.g., 'Bogota' and 'Bogotá')."
            "\n - If you see a nationality, merge it with the country name  (e.g., 'Russian' to 'Russia')."
            "\n - Do not skip any entities in the output; return all the unique entities."
            "\n - Do not include any additional explanations; just output the consolidated list of unique GPE entities."
            "\n - **Do not make any comments or notes; ONLY output the consolidated list.**"
            "\n - **Do not talk about ANY corrections you have done.**"
            "\n - **Do not talk about how you did the corrections.**"
            "\n - **Work only with the GPE entities provided in the input**."
            "\n - **If you do not see any GPE entities in the input, output an empty string**."
            "\n\n\nHere are the entities to process:\n\n"
            "### INPUT\n\n"
            f"{entities_string}\n\n"
            "### OUTPUT:"
        )
        return content


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
