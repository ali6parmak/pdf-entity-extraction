import re
import json
from ollama import Client


def encode_sentences(text):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    encoded = []
    for i, sent in enumerate(sentences, 1):
        encoded.append(f"[SENTENCE_{i}] {sent} [SENTENCE_{i}]")
    return " ".join(encoded)


def decode_sentences(encoded_text):
    pattern = re.compile(r"\[SENTENCE_(\d+)\](.*?)\[SENTENCE_\1\]", re.DOTALL)
    return {int(num): sent.strip() for num, sent in pattern.findall(encoded_text)}


def try_model(model_name: str):
    text_to_analyze = """REPORT No. 121/09 PETITION 1186-04 ADMISSIBILITY OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS) HONDURAS November 12, 2009
    
    I. SUMMARY

    1. On November 5, 2004, the Inter-American Commission on Human Rights (hereinafter "the Inter-American Commission," "the Commission," or "the IACHR") received a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI: Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people of the department of Gracias a Dios (hereinafter "the petitioners") 1, against the State of Honduras (hereinafter "Honduras," "the State," or "the Honduran State"), to the detriment of the divers who are members of the Miskitu people (hereinafter the "alleged victims" or the "Miskitu divers"). The petition alleges that the State has failed to supervise the working conditions of persons who have been and are employed in underwater fishing in the department of Gracias a Dios, who are subject to labor exploitation, which has caused a situation of such proportions and gravity that it endangers the integrity of the Miskitu people, as thousands have suffered multiple and irreversible physical disabilities, and many have died.

    2. In the petition, it is alleged that the State is responsible for violating the fundamental rights of the divers who are members of the Miskitu people established in Articles 4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26 (progressive development of economic, social and cultural rights), in conjunction with Articles 1(1) and 2, all of the American Convention on Human Rights (hereinafter the "Convention" or the "American Convention") and Convention 169 of the International Labor Organization "Concerning Indigenous and Tribal Peoples in Independent Countries" (hereinafter "ILO Convention 169"). As regards the admissibility requirements, they state that they have not had access to domestic remedies, either administrative or judicial, due to their condition of extreme poverty and the failure of the State to provide adequate mechanisms. They state that in those cases in which they have had access to domestic remedies, they were not expeditious or effective, leading to an unwarranted delay in resolving the actions.

    3. The State indicates that it has a specific legal system of protection that regulates labor relations between employers and workers, the procedures to be followed, the institutions, and the competent personnel, so that the persons engaged in underwater fishing can demand respect for and observance of their rights. Moreover, it argues that the cases brought by the persons affected before the competent organs, both administrative and judicial, were not concluded due to omission and abandonment by the petitioners, accordingly they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.
    
    4. This is a fake paragraph to test Inter-American Commission on Human Rights.
    """

    encoded_text = encode_sentences(text_to_analyze)

    prompt = f"""You are a Named Entity Recognition system. Extract ALL entities from the text and return ONLY a JSON array.

Task: Extract entities of these types:
- PERSON: Names of people
- ORGANIZATION: Companies, institutions, agencies
- LOCATION: Cities, countries, geographic locations

Instructions:
1. The text is split into sentences, each wrapped with special tokens: [SENTENCE_1], [SENTENCE_2], etc.
2. For each entity, extract the sentence number from the special token in which it appears.
3. Return ONLY a valid JSON array.
4. Each entity must have: text, type, sentence (the sentence number as an integer, e.g., 1, 2, 3).
5. Do NOT include markdown, explanations, or extra text.

Example output format:
[
  {{"text": "John Doe", "type": "PERSON", "sentence": 1}},
  {{"text": "Google", "type": "ORGANIZATION", "sentence": 1}},
  {{"text": "New York", "type": "LOCATION", "sentence": 2}}
]

Text to analyze:
{encoded_text}

Output (JSON array only):
"""

    client = Client(host="http://localhost:11434")
    response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])

    print("\n" + "=" * 80)
    print("RAW LLM RESPONSE:")
    print("=" * 80)
    raw_response = response["message"]["content"]
    print(raw_response)

    try:
        if raw_response.startswith("```json") and raw_response.endswith("```"):
            raw_response = raw_response[7:-3]
        entities = json.loads(raw_response.strip())
        sent_map = decode_sentences(encoded_text)
        for ent in entities:
            sent_num = ent.get("sentence")
            print(f"Entity: {ent['text']} (Type: {ent['type']}, Sentence: {sent_num})")
            print(f"Sentence: {sent_map.get(sent_num, '')}\n")
            print("-" * 80)
    except Exception as e:
        print("Could not parse entities or map sentences:", e)
        print(entities)


if __name__ == "__main__":
    try_model("gpt-oss")
