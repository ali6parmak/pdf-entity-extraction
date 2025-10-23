from ollama import Client


def try_model(model_name: str):
    prompt = """System role:
    You are a professional Legal NER system. Extract entities from the provided text and return a single valid JSON object that strictly matches the schema below. Do not include any extra commentary.

    Task:
    Given the input text, identify and cluster all mentions of the following entity types:
    - LAW: Named acts/codes/regulations (e.g., “Communications Decency Act”, “Civil Code”).
    - STATUTE: A codified source or named statute family without a specific provision (e.g., “47 U.S.C.”, “Penal Code”).
    - PROVISION: A specific article/section/clause within a LAW or STATUTE (e.g., “47 U.S.C. § 230(c)(1)”, “Article 5(3)”).
    - COURT: Court names (e.g., “United States District Court for the Southern District of New York”).
    - JUDGE: Judicial officers (e.g., “Hon. Jane M. Doe”, “Justice Smith”).
    - LAWYER: Legal counsel/attorneys (e.g., “John R. Smith, Esq.”, “Counsel Mary Lee”).
    - CASE_NUMBER: Docket/case identifiers (e.g., “1:23-cv-12345”, “No. 21-101”).


    Return policy:
    - Return ONLY the JSON object (no markdown, no prose, no trailing commas).
    - If no entities are found, return {"entities": [], "meta": {...}}.

    Output format:
        {
            canonical_name: str,
            entity_type: str,
            source_text: str,
        }

    In the `source_text` field, you should return the exact substring of the input text that in order to find the reference in the original text.

    Here is the text you are going to work on:

    REPORT No. 121/09
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

    client = Client(host="http://localhost:11434")

    response = client.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    print(response["message"]["content"])


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

if __name__ == "__main__":
    try_model("gpt-oss")
    # check_index("American Convention on Human Rights (hereinafter “the Inter-American Commission,” “the Commission,” or “ the IACHR”) received")
