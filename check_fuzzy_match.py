import json
import re
from typing import List, Optional, Tuple
from ollama import Client

from data_model.ExtractedEntity import ExtractedEntity


def find_entity_position_fuzzy(entity_text: str, original_text: str) -> Optional[Tuple[int, int]]:
    start = original_text.find(entity_text)
    if start != -1:
        return (start, start + len(entity_text))

    start = original_text.lower().find(entity_text.lower())
    if start != -1:
        return (start, start + len(entity_text))

    entity_normalized = entity_text.replace("'s", "").replace("'s", "").strip()
    entity_normalized = re.sub(r"\s*-\s*", "-", entity_normalized)
    entity_normalized = re.sub(r"\s+", " ", entity_normalized)

    entity_tokens = entity_normalized.replace("-", " ").split()
    if not entity_tokens:
        return None

    pattern_parts = []
    for i, token in enumerate(entity_tokens):
        escaped_token = re.escape(token)
        pattern_parts.append(escaped_token)
        if i < len(entity_tokens) - 1:
            pattern_parts.append(r"\s*-?\s*")

    pattern = "".join(pattern_parts) + r"(?:\s*'s?)?"

    try:
        match = re.search(pattern, original_text, re.IGNORECASE)
        if match:
            return (match.start(), match.end())
    except re.error:
        pass

    return None


def find_all_entity_positions_fuzzy(entity_text: str, original_text: str) -> List[Tuple[int, int]]:
    positions = []

    entity_normalized = entity_text.replace("'s", "").replace("'s", "").strip()
    entity_normalized = re.sub(r"\s*-\s*", "-", entity_normalized)
    entity_normalized = re.sub(r"\s+", " ", entity_normalized)

    entity_tokens = entity_normalized.replace("-", " ").split()
    if not entity_tokens:
        return positions

    pattern_parts = []
    for i, token in enumerate(entity_tokens):
        escaped_token = re.escape(token)
        pattern_parts.append(escaped_token)
        if i < len(entity_tokens) - 1:
            pattern_parts.append(r"\s*-?\s*")

    pattern = "".join(pattern_parts) + r"(?:\s*'s?)?"

    try:
        for match in re.finditer(pattern, original_text, re.IGNORECASE):
            positions.append((match.start(), match.end()))
    except re.error:
        pass

    if not positions:
        search_text = original_text.lower()
        entity_lower = entity_text.lower()
        start = 0
        while True:
            pos = search_text.find(entity_lower, start)
            if pos == -1:
                break
            positions.append((pos, pos + len(entity_text)))
            start = pos + 1

    return positions


def parse_llm_response(response_text: str, original_text: str, find_all_occurrences: bool = True) -> List[ExtractedEntity]:
    try:
        response_text = response_text.strip()

        if response_text.startswith("```json"):
            response_text = response_text[7:]
        elif response_text.startswith("```"):
            response_text = response_text[3:]

        if response_text.endswith("```"):
            response_text = response_text[:-3]

        response_text = response_text.strip()

        entities_data = json.loads(response_text)

        if isinstance(entities_data, dict) and "entities" in entities_data:
            entities_data = entities_data["entities"]

        if not isinstance(entities_data, list):
            print(f"Warning: Expected list, got {type(entities_data)}")
            return []

        extracted_entities = []

        print("\n" + "=" * 80)
        print("ENTITY OCCURRENCES")
        print("=" * 80)

        for entity in entities_data:
            entity_text = entity.get("text", entity.get("source_text", ""))
            entity_type = entity.get("type", entity.get("entity_type", "")).upper()

            if not entity_text or not entity_type:
                continue

            if find_all_occurrences:
                positions = find_all_entity_positions_fuzzy(entity_text, original_text)
                if positions:
                    print(f"Entity: {entity_text} (Type: {entity_type})")
                    print(f"Positions: {positions}")
                    print("-" * 80)
            else:
                position = find_entity_position_fuzzy(entity_text, original_text)
                positions = [position] if position else []

            if not positions:
                print(f"Warning: Could not find position for entity '{entity_text}'")
                continue

            for start, end in positions:
                actual_text = original_text[start:end]
                extracted_entities.append(
                    ExtractedEntity(text=actual_text, type=entity_type, character_start=start, character_end=end)
                )

        return extracted_entities

    except json.JSONDecodeError as e:
        print(f"JSON parsing error: {e}")
        print(f"Response was: {response_text[:200]}")
        return []
    except Exception as e:
        print(f"Error parsing response: {e}")
        return []


def try_model(model_name: str):
    text_to_analyze = """REPORT No. 121/09 PETITION 1186-04 ADMISSIBILITY OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS) HONDURAS November 12, 2009
    
    I. SUMMARY

    1. On November 5, 2004, the Inter-American Commission on Human Rights (hereinafter "the Inter-American Commission," "the Commission," or "the IACHR") received a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI: Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people of the department of Gracias a Dios (hereinafter "the petitioners") 1, against the State of Honduras (hereinafter "Honduras," "the State," or "the Honduran State"), to the detriment of the divers who are members of the Miskitu people (hereinafter the "alleged victims" or the "Miskitu divers"). The petition alleges that the State has failed to supervise the working conditions of persons who have been and are employed in underwater fishing in the department of Gracias a Dios, who are subject to labor exploitation, which has caused a situation of such proportions and gravity that it endangers the integrity of the Miskitu people, as thousands have suffered multiple and irreversible physical disabilities, and many have died.

    2. In the petition, it is alleged that the State is responsible for violating the fundamental rights of the divers who are members of the Miskitu people established in Articles 4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26 (progressive development of economic, social and cultural rights), in conjunction with Articles 1(1) and 2, all of the American Convention on Human Rights (hereinafter the "Convention" or the "American Convention") and Convention 169 of the International Labor Organization "Concerning Indigenous and Tribal Peoples in Independent Countries" (hereinafter "ILO Convention 169"). As regards the admissibility requirements, they state that they have not had access to domestic remedies, either administrative or judicial, due to their condition of extreme poverty and the failure of the State to provide adequate mechanisms. They state that in those cases in which they have had access to domestic remedies, they were not expeditious or effective, leading to an unwarranted delay in resolving the actions.

    3. The State indicates that it has a specific legal system of protection that regulates labor relations between employers and workers, the procedures to be followed, the institutions, and the competent personnel, so that the persons engaged in underwater fishing can demand respect for and observance of their rights. Moreover, it argues that the cases brought by the persons affected before the competent organs, both administrative and judicial, were not concluded due to omission and abandonment by the petitioners, accordingly they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.
    
    4. This is a fake paragraph to test Inter-American Commission on Human Rights.
    """

    prompt = f"""You are a Named Entity Recognition system. Extract ALL entities from the text and return ONLY a JSON array.
    
Task: Extract entities of these types:
- PERSON: Names of people
- ORGANIZATION: Companies, institutions, agencies
- LOCATION: Cities, countries, geographic locations

Instructions:
1. Find ALL entity mentions in the text
2. Return ONLY a valid JSON array
3. Each entity must have: text, type
4. Do NOT include markdown, explanations, or extra text

Example output format:
[
{{"text": "John Doe", "type": "PERSON"}},
{{"text": "New York", "type": "LOCATION"}}
]

Text to analyze:
{text_to_analyze}


Output (JSON array only):"""

    client = Client(host="http://localhost:11434")

    response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    print("\n" + "=" * 80)
    print("RAW LLM RESPONSE:")
    print("=" * 80)
    print(response["message"]["content"])

    entities = parse_llm_response(response["message"]["content"], text_to_analyze, find_all_occurrences=True)

    print("\n" + "=" * 80)
    print("EXTRACTED ENTITIES WITH POSITIONS:")
    print("=" * 80)

    entities_by_type = {}
    for entity in entities:
        if entity.type not in entities_by_type:
            entities_by_type[entity.type] = []
        entities_by_type[entity.type].append(entity)

    for entity_type in sorted(entities_by_type.keys()):
        print(f"\n{entity_type}:")
        print("-" * 80)
        for entity in entities_by_type[entity_type]:
            print(f"  Text: '{entity.text}'")
            print(f"  Position: [{entity.character_start}:{entity.character_end}]")
            print(
                f"  Context: ...{text_to_analyze[max(0, entity.character_start-20):entity.character_start]}"
                f"[{entity.text}]"
                f"{text_to_analyze[entity.character_end:min(len(text_to_analyze), entity.character_end+20)]}..."
            )
            print()

    print(f"\nTotal entities extracted: {len(entities)}")
    print(f"Unique entity texts: {len(set(e.text for e in entities))}")


if __name__ == "__main__":
    # try_model("llama3.1:8b")
    print(
        find_entity_position_fuzzy(
            "Inter-American Commission",
            """REPORT No. 121/09 PETITION 1186-04 ADMISSIBILITY OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS) HONDURAS November 12, 2009
    
    I. SUMMARY

    1. On November 5, 2004, the Inter-America Commission on Human Rights (hereinafter "the Inter-America Commission," "the Commission," or "the IACHR") received a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI: Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people of the department of Gracias a Dios (hereinafter "the petitioners") 1, against the State of Honduras (hereinafter "Honduras," "the State," or "the Honduran State"), to the detriment of the divers who are members of the Miskitu people (hereinafter the "alleged victims" or the "Miskitu divers"). The petition alleges that the State has failed to supervise the working conditions of persons who have been and are employed in underwater fishing in the department of Gracias a Dios, who are subject to labor exploitation, which has caused a situation of such proportions and gravity that it endangers the integrity of the Miskitu people, as thousands have suffered multiple and irreversible physical disabilities, and many have died.

    2. In the petition, it is alleged that the State is responsible for violating the fundamental rights of the divers who are members of the Miskitu people established in Articles 4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26 (progressive development of economic, social and cultural rights), in conjunction with Articles 1(1) and 2, all of the American Convention on Human Rights (hereinafter the "Convention" or the "American Convention") and Convention 169 of the International Labor Organization "Concerning Indigenous and Tribal Peoples in Independent Countries" (hereinafter "ILO Convention 169"). As regards the admissibility requirements, they state that they have not had access to domestic remedies, either administrative or judicial, due to their condition of extreme poverty and the failure of the State to provide adequate mechanisms. They state that in those cases in which they have had access to domestic remedies, they were not expeditious or effective, leading to an unwarranted delay in resolving the actions.

    3. The State indicates that it has a specific legal system of protection that regulates labor relations between employers and workers, the procedures to be followed, the institutions, and the competent personnel, so that the persons engaged in underwater fishing can demand respect for and observance of their rights. Moreover, it argues that the cases brought by the persons affected before the competent organs, both administrative and judicial, were not concluded due to omission and abandonment by the petitioners, accordingly they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.
    
    4. This is a fake paragraph to test Inter-America Commission on Human Rights.""",
        )
    )
