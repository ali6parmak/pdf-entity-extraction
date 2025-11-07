from time import time

from Levenshtein import ratio
from ollama import Client


def get_content(entity_texts: list[str]):

    entities_string = "\n".join(entity_texts)

    content = (
        "You are an expert in resolving named entities. Your task is to identify unique entities from a list of "
        "similar names. Consider variations in spelling, abbreviations, and partial matches as potential indicators "
        "of the same entity. Use the examples provided to guide your resolution process.\n\n"
        "### EXAMPLE INPUT 1:\n"
        "Alfredo Francisco Brown\n"
        "Alfredo Brown Manister\n"
        "Alfredo Francisco Brown Manister\n\n"
        "### EXAMPLE OUTPUT 1:\n"
        "Alfredo Francisco Brown Manister\n\n\n"
        "### EXAMPLE INPUT 2:\n"
        "the Asociación de Miskitos Hondureños de Buzos Lisiados\n"
        "the Asociación de Misquitos Hondureños de Buzos Lisiados\n\n"
        "### EXAMPLE OUTPUT 2:\n"
        "the Asociación de Miskitos Hondureños de Buzos Lisiados\n\n\n"
        "### EXAMPLE INPUT 3:\n"
        "James Dave King\n"
        "James Dwight Florence\n\n"
        "### EXAMPLE OUTPUT 3:\n"
        "James Dave King\n"
        "James Dwight Florence\n\n\n"
        "\n"
        "Here are the instructions that you should follow:\n"
        "- Compare each pair of entities and determine if they refer to the same entity.\n"
        "- Consider minor spelling variations and additional words as potential matches.\n"
        "- Provide the most complete and accurate version of the entity as the output.\n"
        "- Do not skip any entities in the output, return all the unique entities.\n"
        "- Do not make any additional explanations, just output the entities.\n"
        "\n\n\n"
        "Here are the list of entities you will resolve:\n\n"
        "### INPUT:\n"
        f"{entities_string}"
        "\n\n"
        f"### OUTPUT:\n"
    )
    return content


def get_content_2(entity_texts: list[str]):
    entities_string = "\n".join(entity_texts)
    content = f"""
    You are an expert in data cleaning and entity resolution. Your task is to analyze a list of person names extracted from text data and consolidate entries that refer to the same individual, even if there are minor variations due to typos, accents, or missing names.

**Instructions:**

1. **Group Similar Names:**
   - Identify names that refer to the same person.
   - Consider variations such as spelling differences, missing middle names, or presence of accents (e.g., "Ali" vs. "Alí").

2. **Select the Most Complete Name:**
   - For each group of similar names, choose the most complete and accurate version as the representative name.
   - Prefer names that include full middle names over those that have initials or omit them.

3. **Provide the Consolidated List:**
   - Return a deduplicated list of unique person names.
   - Ensure the list is formatted as a simple, clear list without explanations or additional commentary.

**Examples:**

- "Alfredo Brown Manister", "Alfredo Francisco Brown", and "Alfredo Francisco Brown Manister" should be grouped, and the most complete name "Alfredo Francisco Brown Manister" should be selected.
- "Alí Herrera Ayanco" and "Ali Herrera Ayanco" refer to the same person; choose the name with the correct spelling and accents.

**Here is the list of names to process:**

    {entities_string}
    
    ### OUTPUT:\n
    
    """
    return content


def get_content_3(entity_texts: list[str]):
    entities_string = "\n".join(entity_texts)
    content = """You are an expert in data cleaning and entity resolution. Given a list of person names that may contain duplicates or variations referring to the same individual, please group the names that refer to the same person. For each group, provide the most complete and accurate full name as the canonical name.

Please output the results in the following JSON format:
[
  {
    "canonical_name": "Full Name",
    "aliases": ["Variation1", "Variation2", ...]
  },
  ...
]

Here is the list of person names:
"""
    content += f"\n\n{entities_string}\n\n"
    content += "### OUTPUT:\n"

    return content


def get_person_entity_prompt(entity_texts: list[str]):
    entities_string = "\n".join(entity_texts)
    content = (
        "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of person names "
        "and consolidate entries that refer to the same individual, even if there are minor variations due to typos, accents, or missing names."
        "\n\n"
        "Here are the instructions that you should follow:\n"
        "\n - Identify names that refer to the same person."
        '\n - Consider variations such as spelling differences, missing middle names, or presence of accents (e.g., "Ali" vs. "Alí").'
        "\n - For each group of similar names, choose the most complete and accurate version as the representative name."
        "\n - Prefer names that include full middle names over those that have initials or omit them."
        "\n - Do not skip any entities in the output, return all the unique entities."
        "\n - Do not make any additional explanations, just output the entities."
        "\n\n\nHere are the names to process:\n\n"
        "### INPUT\n\n"
        f"{entities_string}\n\n\n"
        f"### OUTPUT:"
    )
    return content


def get_organization_entity_prompt(entity_texts: list[str]):
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


def get_provision_entity_prompt(entity_texts: list[str]):
    entities_string = "\n".join(entity_texts)
    content = (
        "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of provisions (e.g., articles, sections, responses)  "
        "and extract individual provisions from compound entries. Some entries may list multiple provisions together, and your goal is to separate them into individual entries."
        "\n\n"
        "Here are the instructions that you should follow:\n"
        "\n - Identify all individual provisions mentioned in each entry."
        '\n - Separate compound entries into individual provisions. For example, "Articles 1, 2, and 3" should be split into "Article 1", "Article 2", and "Article 3".'
        ' For example, "Article 10(1)(a) and (b)" should be split into "Article 10(1)(a)" and "Article 10(1)(b)".'
        "\n - Ensure that the provision type (e.g., Article, Section) is correctly associated with each number or letter."
        '\n - Retain any subsection indicators such as numbers or letters in parentheses (e.g., "Article 17(1)").'
        '\n - Normalize the provision type to singular form (e.g., "Articles" becomes "Article").'
        "\n - Correct minor formatting issues, such as extra or missing parentheses. Do not talk about the fix."
        # "\n - Correct minor formatting issues, such as extra or missing parentheses."
        # "\n - There might be minor formatting issues in the input, such as extra or missing parentheses. Please correct them before processing and do not make any additional explanations after the fix." ###
        "\n - Do not skip any provisions; return all individual provisions found in the input."
        # "\n - Do not make any additional explanations; just output the list of individual provisions, each on its own line."
        "\n - Do not make any additional explanations, comments, or clarifications; just output the list of individual provisions, each on its own line."
        "\n\n\nHere are the provisions to process:\n\n"
        "### INPUT\n\n"
        f"{entities_string}\n\n"
        "### OUTPUT:"
    )
    return content


def get_law_entity_prompt(entity_texts: list[str]):
    entities_string = "\n".join(entity_texts)
    content = (
        "You are an expert in legal document analysis and data processing. Your task is to analyze a list of entities extracted from legal texts, "
        "specifically focusing on laws, conventions, treaties, and similar documents (referred to as 'law entities'). "
        "Your goal is to extract and consolidate the unique law entities, ensuring that variations referring to the same law are combined appropriately."
        "\n\n"
        "Here are the instructions you should follow:\n"
        "\n - For any entity that includes a specific article, section, or clause (e.g., 'Article 29 of the French Convention'), extract only the name of the law (e.g., 'French Convention')."
        "\n - Normalize the names by removing leading articles (e.g., 'the'), quotation marks, and unnecessary whitespace."
        "\n - Consider variations in naming that refer to the same law as the same entity. For example, 'French Convention', 'the French Convention', and 'the “French Convention' all refer to 'French Convention'."
        "\n - Consolidate these variations into a single, unique law entity."
        "\n - Ensure that you extract only the names of the laws, conventions, or treaties, without including any articles or sections."
        "\n - Do not make any additional explanations, comments, or clarifications; just output the list of individual laws, each on its own line."
        "\n\n\nHere are the entities to process:\n\n"
        "### INPUT\n\n"
        f"{entities_string}\n\n"
        "### OUTPUT:"
    )
    return content


def get_person_entity_group_prompt(entity_texts: list[str]):
    entities_string = "\n".join(entity_texts)
    content = (
        "You are an expert in data cleaning and entity resolution. Your task is to analyze a list of person names "
        "and group these names according to their similarities with each other. Consider the minor variations due to typos, accents, or missing names."
        "\n\n"
        "Here are the instructions that you should follow:\n"
        "\n - Identify names that refer to the same person."
        '\n - Consider variations such as spelling differences, missing middle names, or presence of accents (e.g., "Ali" vs. "Alí").'
        "\n - Do not skip any entities in the output, return all the unique entities."
        "\n - Do not make any additional explanations, just output the entities."
        "\n\n\nHere are the names to process:\n\n"
        "### INPUT\n\n"
        f"{entities_string}\n\n\n"
        f"### OUTPUT:"
    )
    return content


def get_person_entities():
    # person_entities = ['Alfredo Brown Manister', 'Alfredo Francisco Brown', 'Alfredo Francisco Brown Manister',
    #                    'Alí Herrera Ayanco', 'Amilton Bonaparte Clemente', 'Amilton Clemente Bonaparte',
    #                    'Amisterio Bans Valeriano', 'Amisterio Vans Valeriano', 'Anastacio Richard Bais',
    #                    'Andrés Miranda Clemente', 'Arpin Robles Tatayon', 'Arturo Ribón Avila', 'Bans López Solaisa',
    #                    'Bernardo Blakaus Emos', 'Bernardo Julián Trino', 'Caballero Delgado', 'Carcoth Padmoe Millar',
    #                    'Carcoth Padmoe Miller', 'Carlos Castellón Cárdenas', 'Cendela López Kilton', 'Clare K. Roberts',
    #                    'Cooper Cresencio', 'Crescencio Cooper Jems', 'Daniel Dereck', 'Daniel Dereck Thomas',
    #                    'Daniel Flores Reyes', 'David Esteban Bradley', 'Dereck Claros', 'Durand',
    #                    'Efraín Rosales Kirington',
    #                    'Efraín Rosales Kirrinton', 'Eran Herrera Paulisto', 'Erbacio Martínez',
    #                    'Especel Bradle Valeriano',
    #                    'Evecleto Londres Yumida', 'Evecleto Londres Yumidal', 'Ex', 'Ex Dereck Claro',
    #                    'Ex Dereck Claros',
    #                    'Felipe González', 'Flaviano Martínez', 'Flaviano Martínez López', 'Florentín Melendez',
    #                    'Fredy Federico Salazar', 'Félix Osorio Presby', 'Herrera Paulisto', 'Hildo Ambrocio Trino',
    #                    'Hildo Ambrosio', 'José Martínez López', 'José Marín', 'José Trino Pérez',
    #                    'José Trino Pérez Nacril',
    #                    'Lemus', 'Leonel Saty Méndez', 'Licar Méndez G.', 'Licar Méndez Gutiérrez', 'Londres Yumida',
    #                    'Lorenzo Lemon Bonaparte', 'Luis Felipe Bravo Mena', 'Myrna Mack Chang',
    #                    'Mármol Williams García',
    #                    'Opario Lemoth Morris', 'Paolo G. Carozza', 'Paulino Blakaus Emos', 'Próspero Bendles Marcelino',
    #                    'Ralph Valderramos', 'Ralph Valderramos Álvarez', 'Ramon Allen Felman', 'Ramón Allen Felman',
    #                    'Ramón Allen Ferman', 'Raquel Martín de Mejía', 'Roberto Flores Esteban', 'Roger Alfred Gómez',
    #                    'Róger Gómez Alfred', 'Rómulo Flores Enríquez', 'Rómulo Flores Henríquez',
    #                    'Saipon Richard Toledo',
    #                    'Saipón Richard Toledo', 'Santana', 'Sofía Flores Paulisto', 'Timoteo Lemus Pissaty',
    #                    'Timoteo Lemus Pizzati', 'Timoteo Salazar Zelaya', 'Ugarte', 'Valderramos',
    #                    'Velásquez Rodríguez',
    #                    'Vismar Oracio', 'Víctor E. Abramovich', 'Willy Gómez Pastor']

    person_entities = [
        "Alfredo Brown Manister",
        "Alfredo Francisco Brown",
        "Alfredo Francisco Brown Manister",
        "Alí Herrera Ayanco",
        "Amilton Bonaparte Clemente",
        "Amilton Clemente Bonaparte",
        "Amisterio Bans Valeriano",
        "Amisterio Vans Valeriano",
        "Anastacio Richard Bais",
        "Andrés Miranda Clemente",
        "Arpin Robles Tatayon",
        "Arturo Ribón Avila",
        "Bans López Solaisa",
        "Bernardo Blakaus Emos",
        "Bernardo Julián Trino",
        "Caballero Delgado",
        "Carcoth Padmoe Millar",
        "Carcoth Padmoe Miller",
        "Carlos Castellón Cárdenas",
        "Cendela López Kilton",
    ]

    return person_entities


def get_org_entities():
    org_entities = [
        "AMHBLI",
        "Association of Disabled Honduran Miskitu Divers",
        "Atlántida",
        "CEJIL",
        "Commission",
        "Coordinating Body of Rehabilitation Institutors and Associations of Honduras",
        "Court",
        "Dirección General del Trabajo",
        "Gracias a Dios",
        "IACHR",
        "Juzgado de Trabajo",
        "Juzgados de Trabajo",
        "Labor",
        "Labor Court",
        "Labor Courts",
        "MIMAT",
        "Miskitu",
        "OC-11",
        "Procuraduría",
        "Procuraduría de Trabajo",
        "Procuraduría del Trabajo",
    ]
    return org_entities


def get_provision_entities():
    # provision_entities = ['Article 17(1)', 'Article 17(1))', 'Article 19',  'Article 19)',  'Article 30(2)',
    #                       'Article 32',  'Article 44',  'Article 46(1)(a)',  'Article 46(2)', 'Article 46(2)(c) and (b)',
    #                       'Article 47(b)',  'Article 47(c)',  'Article 6(2)',  'Article 8)',  'Articles 1(1) and 2',
    #                       'Articles 24, 25 and 26)',  'Articles 30 and 37',  'Articles 4 (',  'Articles 4 and 5',
    #                       'Articles 4, 5, 8(1), 17(1), 19, 24, 25',  'Articles 4, 5, 8(1), 17(1), 19, 24, 25, and 26',
    #                       'Articles 4, 5, 8, 17(1), 19, 24, 25',  'Articles 4, 5, 8, 17(1), 19, 24, 25, and 26',
    #                       'Articles 46 and 47',  'Articles 637,638, 641, and 643',  'Articles 8 and 25',
    #                       'Articles 8, 25, and 24',  'articles 46(1)(c) and 47(d)',  'response 32',  'sections (a), (h)']

    provision_entities = [
        "Article 17(1)",
        "Article 17(1))",
        "Article 19",
        "Article 19)",
        "Article 30(2)",
        "Article 32",
        "Article 44",
        "Article 46(1)(a)",
        "Article 46(2)",
        "Article 46(2)(c) and (b)",
        "Article 47(b)",
        "Article 47(c)",
        "Article 6(2)",
        "Article 8)",
        "Articles 1(1) and 2",
        "Articles 24, 25 and 26)",
        "Articles 30 and 37",
        "Articles 4 (",
    ]

    # provision_entities = ['Articles 4 and 5',
    #                       'Articles 4, 5, 8(1), 17(1), 19, 24, 25',  'Articles 4, 5, 8(1), 17(1), 19, 24, 25, and 26',
    #                       'Articles 4, 5, 8, 17(1), 19, 24, 25',  'Articles 4, 5, 8, 17(1), 19, 24, 25, and 26',
    #                       'Articles 46 and 47',  'Articles 637,638, 641, and 643',  'Articles 8 and 25',
    #                       'Articles 8, 25, and 24',  'articles 46(1)(c) and 47(d)',  'response 32',  'sections (a), (h)']

    # provision_entities = ['Article 17(1)', 'Article 17(1))', 'Article 19',  'Article 19)',  'Article 30(2)',
    #                       'Article 32']

    # provision_entities = ['Article 44',  'Article 46(1)(a)',  'Article 46(2)', 'Article 46(2)(c) and (b)',
    #                       'Article 47(b)',  'Article 47(c)']

    # provision_entities = ['Articles 4, 5, 8(1), 17(1), 19, 24, 25, and 26',  'articles 46(1)(c) and 47(d)', 'sections (a), (h)']
    return provision_entities


def get_law_entities():
    law_entities = [
        "/93",
        "26",
        "35 I/A Court H.R.",
        "36 I/A Court H.R.",
        "46",
        "American Convention on Human Rights",
        "Article 26 of the American Convention",
        "Article 26 of the Convention",
        "Article 29 of the American Convention",
        "Article 44 of the American Convention",
        "Article 45 of the Charter of the Organization of American States",
        "Article 46 of the American Convention",
        "Article 46(2)",
        "Article 47(c) of the American Convention",
        "Article 6(2) of the American Convention",
        "Article 669 of the Labor Code of Honduras",
        "Article 7 of",
        "Co nvention 169 of the International Labor Organization “Concerning Indigenous and Tribal Peoples in Independent Countries”",
        "Convention",
        "Convention 169",
        "I/A Court H.R.",
        "ILO Convention 169",
        "Labor Code of Honduras",
        "No.",
        "Regulation",
        "Rules of Procedure",
        "The American Convention",
        "The Labor Code",
        "The Occupational Health and Safety Regulation",
        "its Rules of Procedure",
        "the American Convention",
        "the American Convention on Human Rights",
        "the Commission’s Rules of Procedure",
        "the Constitution of the Republic",
        "the Convention",
        "the Labor Code",
        "the Regulation on Occupational Health and Safety for Underwater Fishing",
        "the Rules of Procedure of the Inter-American Commission on Human Rights",
        "the “American Convention",
        "the “Rules of Procedure",
    ]

    return law_entities


def get_word_intersect_ratio(word1: str, word2: str):
    words1 = set(word1.lower().split())
    words2 = set(word2.lower().split())
    max_len = max(len(words1), len(words2))
    intersection = len(words1 & words2)
    return intersection / max_len


def find_unique_entities(entity_texts: list[str]):
    non_unique_entity_indexes = []
    for current_text_index, text_1 in enumerate(entity_texts):
        if current_text_index in non_unique_entity_indexes:
            continue
        for comparison_text_index, text_2 in enumerate(entity_texts):
            if current_text_index == comparison_text_index:
                continue
            if comparison_text_index in non_unique_entity_indexes:
                continue
            if ratio(text_1, text_2) > 0.79 or get_word_intersect_ratio(text_1, text_2) > 0.65:
                if current_text_index not in non_unique_entity_indexes:
                    non_unique_entity_indexes.append(current_text_index)
                non_unique_entity_indexes.append(comparison_text_index)
                continue

    non_unique_entities = [entity_texts[index] for index in non_unique_entity_indexes]
    unique_entities = [entity_texts[index] for index in range(len(entity_texts)) if index not in non_unique_entity_indexes]
    print(non_unique_entity_indexes)
    print("NON UNIQUE:", non_unique_entities)
    print("UNIQUE:", unique_entities)
    return sorted(non_unique_entities), sorted(unique_entities)


if __name__ == "__main__":

    # extractor = MultipleEntityExtractor()
    # start = time()
    # entities_dict = extractor.extract_entities("cejil_staging33")
    # print("Extraction finished in", round(time() - start, 2), "seconds")
    # sorted_person_entities = sorted(entities_dict["PERSON"].items())
    # person_entities = [entity_text for entity_text, data in sorted_person_entities]
    person_entities = get_person_entities()
    org_entities = get_org_entities()
    provision_entities = get_provision_entities()
    law_entities = get_law_entities()

    non_unique_entities, unique_entities = find_unique_entities(person_entities)
    # non_unique_entities, unique_entities = find_unique_entities(org_entities)
    # non_unique_entities, unique_entities = find_unique_entities(provision_entities)

    client = Client(host=f"http://localhost:11434")
    # content = get_person_entity_prompt(non_unique_entities)
    # content = get_person_entity_prompt(unique_entities)
    content = get_person_entity_group_prompt(person_entities)

    # content = get_organization_entity_prompt(non_unique_entities)

    # content = get_provision_entity_prompt(non_unique_entities)
    # content = get_provision_entity_prompt(provision_entities)

    # content = get_law_entity_prompt(law_entities)

    start = time()
    # options = {"temperature": 0.2, "top_k": 5}
    options = {"temperature": 0}
    # response = client.chat(model="gemma2:9b", messages=[{"role": "user", "content": content}])
    # response = client.chat(model="llama3.1", messages=[{"role": "user", "content": content}])
    response = client.chat(model="llama3.1", options=options, messages=[{"role": "user", "content": content}])
    print("Chat response finished in", round(time() - start, 2), "seconds")
    response_content = response["message"]["content"]
    print("\033[94m" + response_content + "\033[0m")
