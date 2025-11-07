from ollama import Client
import json


def try_model(model_name: str):
    segments = [
        """
    REPORT No. 121/09
    PETITION 1186-04
    ADMISSIBILITY
    OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS)
    HONDURAS
    November 12, 2009""",
        """I. SUMMARY""",
        """1. On November 5, 2004, the Inter-American Commission on Human Rights (hereinafter “the Inter-American Commission,” “the Commission,” or “the IACHR”) received a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI: Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people of the department of Gracias a Dios (hereinafter “the petitioners”) 1, against the State of Honduras (hereinafter “Honduras,” “the State,” or “the Honduran State”), to the detriment of the divers who are members of the Miskitu people (hereinafter the “alleged victims” or the “Miskitu divers”). The petition alleges that the State has failed to supervise the working conditions of persons who have been and are employed in underwater fishing in the department of Gracias a Dios, who are subject to labor exploitation, which has caused a situation of such proportions and gravity that it endangers the integrity of the Miskitu people, as thousands have suffered multiple and irreversible physical disabilities, and many have died.""",
        """2. In the petition, it is alleged that the State is responsible for violating the fundamental rights of the divers who are members of the Miskitu people established in Articles 4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26 (progressive development of economic, social and cultural rights), in conjunction with Articles 1(1) and 2, all of the American Convention on Human Rights (hereinafter the “Convention” or the “American Convention”) and Convention 169 of the International Labor Organization “Concerning Indigenous and Tribal Peoples in Independent Countries” (hereinafter “ILO Convention 169”). As regards the admissibility requirements, they state that they have not had access to domestic remedies, either administrative or judicial, due to their condition of extreme poverty and the failure of the State to provide adequate mechanisms. They state that in those cases in which they have had access to domestic remedies, they were not expeditious or effective, leading to an unwarranted delay in resolving the actions.""",
        """3. The State indicates that it has a specific legal system of protection that regulates labor relations between employers and workers, the procedures to be followed, the institutions, and the competent personnel, so that the persons engaged in underwater fishing can demand respect for and observance of their rights. Moreover, it argues that the cases brought by the persons affected before the competent organs, both administrative and judicial, were not concluded due to omission and abandonment by the petitioners, accordingly they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.""",
    ]
    base_prompt = """You are a Named Entity Recognition system. Extract ALL entities from the text and return ONLY a JSON array.
    
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
"""

    result = []

    for segment in segments:
        prompt = base_prompt.format(text_to_analyze=segment)

        client = Client(host="http://localhost:11434")

        response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
        print(response["message"]["content"])
        result.extend(json.loads(response["message"]["content"]))

    print("*" * 80)
    print(result)
    print("*" * 80)

    full_text = "\n".join(segments).lower()
    for entity in result:
        if entity["text"].lower() not in full_text:
            print(f"Entity {entity['text']} not found in full text")
            print(f"Entity: {entity['text']}")
            # print(f"Full text: {full_text}")
            print("*" * 80)

    return result


def check_index(search_text: str):
    text = """REPORT No. 121/09
    PETITION 1186-04
    ADMISSIBILITY
    OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS)
    HONDURAS
    November 12, 2009
    I.
    SUMMARY
    1.
    On November 5, 2004, the Inter-American Commission on Human Rights
    (hereinafter “the Inter-American Commission,” “the Commission,” or “the IACHR”) received
    a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI:
    Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu
    Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla
    Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela
    López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people
    of the department of Gracias a Dios (hereinafter “the petitioners”) 1, against the State of
    Honduras (hereinafter “Honduras,” “the State,” or “the Honduran State”), to the detriment of
    the divers who are members of the Miskitu people (hereinafter the “alleged victims” or the
    “Miskitu divers”). The petition alleges that the State has failed to supervise the working
    conditions of persons who have been and are employed in underwater fishing in the
    department of Gracias a Dios, who are subject to labor exploitation, which has caused a
    situation of such proportions and gravity that it endangers the integrity of the Miskitu people,
    as thousands have suffered multiple and irreversible physical disabilities, and many have
    died.
    2.
    In the petition, it is alleged that the State is responsible for violating the
    fundamental rights of the divers who are members of the Miskitu people established in Articles
    4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the
    family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26
    (progressive development of economic, social and cultural rights), in conjunction with Articles
    1(1) and 2, all of the American Convention on Human Rights (hereinafter the “Convention” or
    the “American Convention”) and Convention 169 of the International Labor Organization
    “Concerning Indigenous and Tribal Peoples in Independent Countries” (hereinafter “ILO
    Convention 169”). As regards the admissibility requirements, they state that they have not
    had access to domestic remedies, either administrative or judicial, due to their condition of
    extreme poverty and the failure of the State to provide adequate mechanisms. They state
    that in those cases in which they have had access to domestic remedies, they were not
    expeditious or effective, leading to an unwarranted delay in resolving the actions.
    3.
    The State indicates that it has a specific legal system of protection that
    regulates labor relations between employers and workers, the procedures to be followed, the
    institutions, and the competent personnel, so that the persons engaged in underwater fishing
    can demand respect for and observance of their rights. Moreover, it argues that the cases
    brought by the persons affected before the competent organs, both administrative and
    judicial, were not concluded due to omission and abandonment by the petitioners, accordingly
    they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.
    """

    print("Start index: ", text.index(search_text))
    print("End index: ", text.index(search_text) + len(search_text))


def check_sample(model_name: str):
    sample = """The Presidential Advisory Group met again on November 26, and discussed a proposal, made earlier by Public Television System chairman Wu Feng - shan and already favored by most members of the Group, for " one China " to be addressed through " three acknowledgements and four suggestions. " A firm consensus was finally arrived at, and the Group submitted to the president its proposal that Taiwan should " respond to the mainland \'s \' one China \' proposition in accordance with the constitution of the ROC. " The government and opposition parties continued to disagree regarding this consensus. DPP legislator Lin Cho - shui described it as " acceptable, if not very satisfactory. " PFP spokesman Li Ching - an called it " a conclusion without a conclusion. """

    sample_2 = """They also provided 190 million US dollars of installation insurance for a national key project: the Lanhua Chemical Fertilizer Plant reform and expansion project. Aiming at the development requirements of the Gansu tourism industry, People \'s Insurance Co. actively promotes travel insurance for overseas tourists, and took the lead at home in providing insurance for individual overseas tourists, which made sure that all those who came sightseeing in Gansu Province during the " eighth five - year plan " period had insurance. Gansu Province also actively explored high risk business. During the " eighth five - year plan " period, it participated in the co-insurance of satellite launching, with a shared risk amount reaching 10 million yuan, and, paying 5 million yuan in indemnity, became the northwest \'s first company to participate in the aerospace industry."""

    base_prompt = """You are a Named Entity Recognition system. Extract ALL entities from the text and return ONLY a JSON array.

**Do not care about the context of the whole text, only focus on extracting entities.**
    
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
{sample}
"""

    prompt = base_prompt.format(sample=sample)

    client = Client(host="http://localhost:11434")
    response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    print(response["message"]["content"])

    # [{'text': 'The Presidential Advisory Group', 'type': 'ORGANIZATION', 'start': 0, 'end': 31},
    # {'text': 'Public Television System', 'type': 'ORGANIZATION', 'start': 100, 'end': 124},
    # {'text': 'Wu Feng - shan', 'type': 'PERSON', 'start': 134, 'end': 148},
    # {'text': 'Group', 'type': 'ORGANIZATION', 'start': 192, 'end': 197},
    # {'text': 'China', 'type': 'LOCATION', 'start': 209, 'end': 214},
    # {'text': 'Group', 'type': 'ORGANIZATION', 'start': 339, 'end': 344},
    # {'text': 'Taiwan', 'type': 'LOCATION', 'start': 390, 'end': 396},
    # {'text': 'China', 'type': 'LOCATION', 'start': 439, 'end': 444},
    # {'text': 'ROC', 'type': 'LOCATION', 'start': 502, 'end': 505},
    # {'text': 'DPP', 'type': 'ORGANIZATION', 'start': 595, 'end': 598},
    # {'text': 'Lin Cho - shui', 'type': 'PERSON', 'start': 610, 'end': 624},
    # {'text': 'PFP', 'type': 'ORGANIZATION', 'start': 683, 'end': 686},
    # {'text': 'Li Ching - an', 'type': 'PERSON', 'start': 697, 'end': 710}]

    prompt_2 = base_prompt.format(sample=sample_2)
    response_2 = client.chat(model=model_name, messages=[{"role": "user", "content": prompt_2}])
    print(response_2["message"]["content"])

    # {'text': 'Gansu', 'type': 'LOCATION', 'start': 209, 'end': 214}


if __name__ == "__main__":
    # try_model("gpt-oss")
    check_sample("gpt-oss")
    # check_sample("deepseek-r1:14b")
    # check_index("American Convention on Human Rights (hereinafter “the Inter-American Commission,” “the Commission,” or “ the IACHR”) received")
