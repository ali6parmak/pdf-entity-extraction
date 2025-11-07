from ollama import Client


def try_model(model_name: str):
    # text_to_analyze = """REPORT No. 121/09 PETITION 1186-04 ADMISSIBILITY OPARIO LEMOTH MORRIS ET AL. (MISKITU DIVERS) HONDURAS November 12, 2009

    # I. SUMMARY

    # 1. On November 5, 2004, the Inter-American Commission on Human Rights (hereinafter "the Inter-American Commission," "the Commission," or "the IACHR") received a complaint submitted by the Asociación de Miskitos Hondureños de Buzos Lisiados (AMHBLI: Association of Disabled Honduran Miskitu Divers); the Asociación de Mujeres Miskitas Miskitu Indian Mairin Asla Takanka (MIMAT: Association of Miskitu Women); and the Almuk Nani Asla Takanka Council of Elders, respectively, represented by Arquímedes García López, Cendela López Kilton, and Bans López Solaisa, all in representation of the Miskitu indigenous people of the department of Gracias a Dios (hereinafter "the petitioners") 1, against the State of Honduras (hereinafter "Honduras," "the State," or "the Honduran State"), to the detriment of the divers who are members of the Miskitu people (hereinafter the "alleged victims" or the "Miskitu divers"). The petition alleges that the State has failed to supervise the working conditions of persons who have been and are employed in underwater fishing in the department of Gracias a Dios, who are subject to labor exploitation, which has caused a situation of such proportions and gravity that it endangers the integrity of the Miskitu people, as thousands have suffered multiple and irreversible physical disabilities, and many have died.

    # 2. In the petition, it is alleged that the State is responsible for violating the fundamental rights of the divers who are members of the Miskitu people established in Articles 4 (right to life), 5 (humane treatment), 8(1) (judicial guarantees), 17(1) (protection of the family), 19 (rights of the child), 24 (equality before the law), 25 (judicial protection), and 26 (progressive development of economic, social and cultural rights), in conjunction with Articles 1(1) and 2, all of the American Convention on Human Rights (hereinafter the "Convention" or the "American Convention") and Convention 169 of the International Labor Organization "Concerning Indigenous and Tribal Peoples in Independent Countries" (hereinafter "ILO Convention 169"). As regards the admissibility requirements, they state that they have not had access to domestic remedies, either administrative or judicial, due to their condition of extreme poverty and the failure of the State to provide adequate mechanisms. They state that in those cases in which they have had access to domestic remedies, they were not expeditious or effective, leading to an unwarranted delay in resolving the actions.

    # 3. The State indicates that it has a specific legal system of protection that regulates labor relations between employers and workers, the procedures to be followed, the institutions, and the competent personnel, so that the persons engaged in underwater fishing can demand respect for and observance of their rights. Moreover, it argues that the cases brought by the persons affected before the competent organs, both administrative and judicial, were not concluded due to omission and abandonment by the petitioners, accordingly they ask that the petition be found inadmissible due to failure to exhaust domestic remedies.

    # 4. This is a fake paragraph to test Inter-American Commission on Human Rights.
    # """

    #     text_to_analyze = """Maria Rodriguez visited the Louvre Museum in Paris, France, on Wednesday, July 12, 2023.

    # The full name of this person is Maria Diaz Rodriguez.

    # It can also be written as M.D. Rodriguez.

    # She is working in an organization called HURIDOCS.

    # The Senate passed Resolution No. 122, establishing a set of rules for the impeachment trial.

    # Maria D. Rodriguez's birthday is July 12, 1980.
    # """

    #     text_to_analyze = """Document 1: Project Overview
    # 1. Introduction

    # This document outlines the scope and phases of Project Chimera, an initiative aimed at streamlining cross-departmental data synchronization. The primary goal is to enhance data integrity and accessibility across our organization. The success of Project Chimera heavily relies on accurate data processing and efficient workflow management, as further elaborated in later sections.

    # 2. Phase 1: Data Collection

    # The initial phase focused on identifying and acquiring relevant datasets from various internal and external sources. We faced several challenges in data harmonization during this period. For detailed specifications regarding specific data sources, especially Data Source A, please refer to Document 2, Section "Data Source A".

    # 3. Phase 2: Analysis

    # Once data collection was complete, the next step involved extensive analysis. This phase required specialized tools and significant computational resources to process the raw information. Our analytical approach is built upon the foundational work described in the preceding "Phase 1: Data Collection" section. The interpretation of these results will be covered in the "Results Interpretation" section below.

    # 4. Results Interpretation

    # The analysis yielded several significant findings, indicating both expected outcomes and some surprising correlations within the data. These findings will inform our strategic decisions moving forward.

    # A more granular breakdown of these observations and their implications is available in

    # Document 3, Section "Detailed Findings".

    # 5. Conclusion

    # Project Chimera has successfully laid the groundwork for improved data practices. The insights gained from our extensive work provide a solid foundation for future development and optimization efforts."""

    text_to_analyze = """A. J. van der Meer, the Chief Scientist at Quantum Innovations International, visited San Francisco for a summit. Anna Johanna van der Meer later presented groundbreaking research. The summit, organized by Quantum Innovations International, attracted experts from Berlin and was featured in The Global Science Review. Dr. van der Meer’s insights were highly praised."""

    prompt = f"""You are a Named Entity Recognition system. Extract ALL entities from the text and return ONLY a JSON array.

Task: Extract entities of these types:
- PERSON: Names of people
- ORGANIZATION: Companies, institutions, agencies
- LOCATION: Cities, countries, geographic locations

Instructions:
1. Find ALL entity mentions in the text.
2. For each entity mention, include:
   - text: the exact mention as it appears in the text
   - type: the entity type
   - canonical_name: the canonical (full, unambiguous) name for the entity. If multiple mentions refer to the same entity (e.g., "M. A. Tresha" and "Morris Ariana Tresha"), use the most complete or widely recognized form as the canonical name.
3. Return ONLY a valid JSON array.
4. Do NOT include markdown, explanations, or extra text.

Example output format:
[
  {{"text": "M. A. Tresha", "type": "PERSON", "canonical_name": "Morris Ariana Tresha"}},
  {{"text": "Morris A. Tresha", "type": "PERSON", "canonical_name": "Morris Ariana Tresha"}},
  {{"text": "Morris Ariana Tresha", "type": "PERSON", "canonical_name": "Morris Ariana Tresha"}},
  {{"text": "New York", "type": "LOCATION", "canonical_name": "New York"}}
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


if __name__ == "__main__":
    # try_model("llama3.1:8b")
    try_model("gpt-oss")
