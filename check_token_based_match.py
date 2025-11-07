import json
import re
from typing import List, Dict
from difflib import SequenceMatcher

from ollama import Client


def tokenize_with_positions(text: str) -> List[Dict]:
    pattern = r"\b\w+\b"
    tokens = []
    for match in re.finditer(pattern, text):
        tokens.append({"token": match.group().lower(), "start": match.start(), "end": match.end()})
    return tokens


def normalize_tokens(text: str) -> List[str]:
    pattern = r"\b\w+\b"
    return [m.group().lower() for m in re.finditer(pattern, text)]


def find_token_sequence(text_tokens: List[str], entity_tokens: List[str], threshold: float = 0.8) -> List[int]:
    matches = []
    entity_len = len(entity_tokens)

    if entity_len == 0:
        return matches

    for i in range(len(text_tokens) - entity_len + 1):
        window = text_tokens[i : i + entity_len]

        similarity = SequenceMatcher(None, window, entity_tokens).ratio()

        if similarity >= threshold:
            matches.append(i)

    return matches


def find_entity_spans_token_based(text: str, entity: str, threshold: float = 0.8) -> List[Dict]:
    text_tokens_with_pos = tokenize_with_positions(text)
    text_tokens = [t["token"] for t in text_tokens_with_pos]
    entity_tokens = normalize_tokens(entity)
    token_matches = find_token_sequence(text_tokens, entity_tokens, threshold)
    results = []
    for token_idx in token_matches:
        start_char = text_tokens_with_pos[token_idx]["start"]
        end_token_idx = token_idx + len(entity_tokens) - 1
        end_char = text_tokens_with_pos[end_token_idx]["end"]

        matched_text = text[start_char:end_char]

        window = text_tokens[token_idx : token_idx + len(entity_tokens)]
        similarity = SequenceMatcher(None, window, entity_tokens).ratio()

        results.append({"text": matched_text, "start": start_char, "end": end_char, "similarity": similarity})

    return results


def process_llm_entities(text: str, entities: List[Dict], threshold: float = 0.8) -> List[Dict]:
    all_mentions = []

    for entity in entities:
        entity_text = entity["text"]
        entity_type = entity["type"]

        spans = find_entity_spans_token_based(text, entity_text, threshold)

        for span in spans:
            all_mentions.append(
                {
                    "text": span["text"],
                    "type": entity_type,
                    "start": span["start"],
                    "end": span["end"],
                    "similarity": span["similarity"],
                    "original_entity": entity_text,
                }
            )

    all_mentions.sort(key=lambda x: x["start"])

    return all_mentions


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

    llm_output = response["message"]["content"]

    json_match = re.search(r"\[.*\]", llm_output, re.DOTALL)
    if json_match:
        entities = json.loads(json_match.group())
    else:
        entities = json.loads(llm_output)

    mentions = process_llm_entities(text_to_analyze, entities, threshold=0.8)

    print(f"\nFound {len(mentions)} entity mentions:")
    for m in mentions:
        print(f"{m['type']:15} | {m['start']:5}-{m['end']:5} | {m['text'][:50]}")

    return mentions


if __name__ == "__main__":
    try_model("llama3.1:8b")
