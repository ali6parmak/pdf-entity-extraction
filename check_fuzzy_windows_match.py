from ollama import Client
from typing import List, Dict
from dataclasses import dataclass
import json
import re
from difflib import SequenceMatcher


@dataclass
class EntityMatch:
    entity_text: str
    matched_text: str
    start_index: int
    end_index: int
    entity_type: str
    similarity_score: float


def normalize_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def calculate_similarity(str1: str, str2: str) -> float:
    return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()


def fuzzy_match_with_window(
    source_text: str,
    entity_text: str,
    entity_type: str,
    min_similarity: float = 0.85,
    window_size_multiplier: float = 2.5,
    min_window_size: int = 50,
    max_window_size: int = 500,
    step_size_ratio: float = 0.5,
) -> List[EntityMatch]:
    normalized_entity = normalize_text(entity_text)
    entity_len = len(normalized_entity)

    window_size = int(entity_len * window_size_multiplier)
    window_size = max(min_window_size, min(window_size, max_window_size))

    step_size = max(1, int(window_size * step_size_ratio))

    matches = []
    source_len = len(source_text)

    matched_ranges = []

    for i in range(0, source_len - entity_len + 1, step_size):
        window_start = max(0, i - window_size // 2)
        window_end = min(source_len, i + entity_len + window_size // 2)
        window_text = source_text[window_start:window_end]

        best_match = None
        best_score = 0
        best_position = -1

        for j in range(len(window_text) - entity_len + 1):
            candidate = window_text[j : j + entity_len]
            score = calculate_similarity(normalized_entity, normalize_text(candidate))

            if score > best_score:
                best_score = score
                best_match = candidate
                best_position = window_start + j

        for length_delta in range(-min(20, entity_len // 4), min(20, entity_len // 4) + 1):
            adjusted_len = entity_len + length_delta
            if adjusted_len < 3:
                continue

            for j in range(len(window_text) - adjusted_len + 1):
                candidate = window_text[j : j + adjusted_len]
                score = calculate_similarity(normalized_entity, normalize_text(candidate))

                if score > best_score:
                    best_score = score
                    best_match = candidate
                    best_position = window_start + j

        if best_score >= min_similarity and best_match and best_position >= 0:
            is_overlapping = False
            for start, end in matched_ranges:
                if not (best_position >= end or best_position + len(best_match) <= start):
                    is_overlapping = True
                    break

            if not is_overlapping:
                match = EntityMatch(
                    entity_text=entity_text,
                    matched_text=best_match,
                    start_index=best_position,
                    end_index=best_position + len(best_match),
                    entity_type=entity_type,
                    similarity_score=best_score,
                )
                matches.append(match)
                matched_ranges.append((best_position, best_position + len(best_match)))

    matches.sort(key=lambda x: x.start_index)

    final_matches = []
    for match in matches:
        is_duplicate = False
        for existing in final_matches:
            overlap_start = max(match.start_index, existing.start_index)
            overlap_end = min(match.end_index, existing.end_index)
            overlap_len = max(0, overlap_end - overlap_start)

            if overlap_len > min(len(match.matched_text), len(existing.matched_text)) * 0.5:
                is_duplicate = True
                break

        if not is_duplicate:
            final_matches.append(match)

    return final_matches


def extract_entities_with_positions(
    source_text: str, entities: List[Dict[str, str]], min_similarity: float = 0.85
) -> List[EntityMatch]:
    all_matches = []

    for entity in entities:
        entity_text = entity.get("text", "")
        entity_type = entity.get("type", "UNKNOWN")

        if not entity_text:
            continue

        matches = fuzzy_match_with_window(
            source_text=source_text, entity_text=entity_text, entity_type=entity_type, min_similarity=min_similarity
        )

        all_matches.extend(matches)

    all_matches.sort(key=lambda x: x.start_index)

    return all_matches


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

    try:
        response_text = response["message"]["content"].strip()

        if response_text.startswith("```"):
            lines = response_text.split("\n")
            json_lines = []
            in_json = False
            for line in lines:
                if line.strip().startswith("```"):
                    in_json = not in_json
                    continue
                if in_json or (line.strip().startswith("[") or line.strip().startswith("{")):
                    json_lines.append(line)
            response_text = "\n".join(json_lines)

        entities = json.loads(response_text)

        print("\n" + "=" * 80)
        print(f"EXTRACTED {len(entities)} ENTITIES:")
        print("=" * 80)
        for i, entity in enumerate(entities, 1):
            print(f"{i}. {entity.get('text', 'N/A')} ({entity.get('type', 'N/A')})")

        print("\n" + "=" * 80)
        print("FINDING ENTITY POSITIONS (with fuzzy matching):")
        print("=" * 80)

        matches = extract_entities_with_positions(source_text=text_to_analyze, entities=entities, min_similarity=0.85)

        print(f"\nFound {len(matches)} total matches in the text:\n")

        for i, match in enumerate(matches, 1):
            print(f"{i}. Entity: '{match.entity_text}'")
            print(f"   Type: {match.entity_type}")
            print(f"   Matched: '{match.matched_text}'")
            print(f"   Position: [{match.start_index}:{match.end_index}]")
            print(f"   Similarity: {match.similarity_score:.2%}")

            context_start = max(0, match.start_index - 30)
            context_end = min(len(text_to_analyze), match.end_index + 30)
            context = text_to_analyze[context_start:context_end]

            match_start_in_context = match.start_index - context_start
            match_end_in_context = match.end_index - context_start

            context_display = (
                context[:match_start_in_context]
                + f">>>{context[match_start_in_context:match_end_in_context]}<<<"
                + context[match_end_in_context:]
            )
            print(f"   Context: ...{context_display}...")
            print()

        print("\n" + "=" * 80)
        print("SUMMARY BY ENTITY:")
        print("=" * 80)

        entity_groups = {}
        for match in matches:
            key = (match.entity_text, match.entity_type)
            if key not in entity_groups:
                entity_groups[key] = []
            entity_groups[key].append(match)

        for (entity_text, entity_type), entity_matches in entity_groups.items():
            print(f"\n'{entity_text}' ({entity_type}): {len(entity_matches)} occurrence(s)")
            for match in entity_matches:
                print(
                    f"  - Position [{match.start_index}:{match.end_index}]: '{match.matched_text}' (similarity: {match.similarity_score:.2%})"
                )

    except json.JSONDecodeError as e:
        print(f"\n[ERROR] Failed to parse JSON response: {e}")
    except Exception as e:
        print(f"\n[ERROR] An error occurred: {e}")
        import traceback

        traceback.print_exc()


def demo_with_test_entities():
    text_to_analyze = """REPORT No. 121/09 PETITION 1186-04 ADMISSIBILITY OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS) HONDURAS November 12, 2009
  
    I. SUMMARY

    1. On November 5, 2004, the Inter-American Commission on Human Rights (hereinafter "the Inter-American Commission," "the Commission," or "the IACHR") received a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI: Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people of the department of Gracias a Dios (hereinafter "the petitioners") 1, against the State of Honduras (hereinafter "Honduras," "the State," or "the Honduran State"), to the detriment of the divers who are members of the Miskitu people (hereinafter the "alleged victims" or the "Miskitu divers"). The petition alleges that the State has failed to supervise the working conditions of persons who have been and are employed in underwater fishing in the department of Gracias a Dios, who are subject to labor exploitation, which has caused a situation of such proportions and gravity that it endangers the integrity of the Miskitu people, as thousands have suffered multiple and irreversible physical disabilities, and many have died.

    2. In the petition, it is alleged that the State is responsible for violating the fundamental rights of the divers who are members of the Miskitu people established in Articles 4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26 (progressive development of economic, social and cultural rights), in conjunction with Articles 1(1) and 2, all of the American Convention on Human Rights (hereinafter the "Convention" or the "American Convention") and Convention 169 of the International Labor Organization "Concerning Indigenous and Tribal Peoples in Independent Countries" (hereinafter "ILO Convention 169"). As regards the admissibility requirements, they state that they have not had access to domestic remedies, either administrative or judicial, due to their condition of extreme poverty and the failure of the State to provide adequate mechanisms. They state that in those cases in which they have had access to domestic remedies, they were not expeditious or effective, leading to an unwarranted delay in resolving the actions.

    3. The State indicates that it has a specific legal system of protection that regulates labor relations between employers and workers, the procedures to be followed, the institutions, and the competent personnel, so that the persons engaged in underwater fishing can demand respect for and observance of their rights. Moreover, it argues that the cases brought by the persons affected before the competent organs, both administrative and judicial, were not concluded due to omission and abandonment by the petitioners, accordingly they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.
  
    4. This is a fake paragraph to test Inter-American Commission on Human Rights.
    """

    test_entities = [
        {"text": "Inter-American Commission on Human Rights", "type": "ORGANIZATION"},
        {"text": "IACHR", "type": "ORGANIZATION"},
        {"text": "Honduras", "type": "LOCATION"},
        {"text": "Arquímedes García López", "type": "PERSON"},
        {"text": "Cendela López Kilton", "type": "PERSON"},
        {"text": "Gracias a Dios", "type": "LOCATION"},
        {"text": "Miskitu people", "type": "ORGANIZATION"},
        {"text": "American Convention on Human Rights", "type": "ORGANIZATION"},
        {"text": "International Labor Organization", "type": "ORGANIZATION"},
        {"text": "the State", "type": "ORGANIZATION"},
    ]

    print("=" * 80)
    print("DEMO: Fuzzy Matching with Test Entities")
    print("=" * 80)
    print(f"\nTesting with {len(test_entities)} predefined entities")
    print("\nEntities to find:")
    for i, entity in enumerate(test_entities, 1):
        print(f"{i}. {entity['text']} ({entity['type']})")

    print("\n" + "=" * 80)
    print("FINDING ENTITY POSITIONS:")
    print("=" * 80)

    matches = extract_entities_with_positions(source_text=text_to_analyze, entities=test_entities, min_similarity=0.85)

    print(f"\nFound {len(matches)} total matches in the text:\n")

    for i, match in enumerate(matches, 1):
        print(f"{i}. Entity: '{match.entity_text}'")
        print(f"   Type: {match.entity_type}")
        print(f"   Matched: '{match.matched_text}'")
        print(f"   Position: [{match.start_index}:{match.end_index}]")
        print(f"   Similarity: {match.similarity_score:.2%}")
        print()

    print("\n" + "=" * 80)
    print("SUMMARY BY ENTITY (showing multiple occurrences):")
    print("=" * 80)

    entity_groups = {}
    for match in matches:
        key = (match.entity_text, match.entity_type)
        if key not in entity_groups:
            entity_groups[key] = []
        entity_groups[key].append(match)

    for (entity_text, entity_type), entity_matches in entity_groups.items():
        print(f"\n'{entity_text}' ({entity_type}): {len(entity_matches)} occurrence(s)")
        for idx, match in enumerate(entity_matches, 1):
            print(
                f"  {idx}. Position [{match.start_index}:{match.end_index}]: '{match.matched_text}' (similarity: {match.similarity_score:.2%})"
            )


if __name__ == "__main__":
    try_model("llama3.1:8b")
