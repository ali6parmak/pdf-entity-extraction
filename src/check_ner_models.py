import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification, AutoConfig
from transformers import pipeline

from configuration import CACHED_DATA_PATH


def aggregate_entities(ner_results: list[dict]):
    entities = []
    current_entity = None

    for entity_dict in ner_results:
        entity_label = entity_dict["entity"]
        if "-" in entity_label:
            dash_index = entity_label.index("-")
            entity_label = entity_label[dash_index + 1 :]
        text = entity_dict["word"].replace("▁", " ").strip()

        if current_entity is None:
            current_entity = {
                "entity_label": entity_label,
                "text": text,
                "start_index": entity_dict["start"],
                "end_index": entity_dict["end"],
            }
        elif entity_dict["start"] == current_entity["end_index"] and entity_label == current_entity["entity_label"]:
            current_entity["text"] += text
            current_entity["end_index"] = entity_dict["end"]
        elif (
            entity_dict["start"] == current_entity["end_index"] + 1
            and entity_label == current_entity["entity_label"]
            and entity_dict["entity"].startswith("I-")
        ):
            current_entity["text"] += " " + text
            current_entity["end_index"] = entity_dict["end"]
        else:
            # End the current entity and start a new one
            entities.append(current_entity)
            current_entity = {
                "entity_label": entity_label,
                "text": text,
                "start_index": entity_dict["start"],
                "end_index": entity_dict["end"],
            }

    # Add the last entity if it exists
    if current_entity:
        entities.append(current_entity)

    return entities


def check_bert_base_ner():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    config = AutoConfig.from_pretrained("dslim/bert-base-NER")
    print(config.id2label)

    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, "example.json").read_text())
    # all_text = "\n".join([segment["text"] for segment in segment_boxes])
    all_text = segment_boxes[0]["text"]
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "My name is Wolfgang and I live in New York"
    ner_results = nlp(all_text)
    print(ner_results)


def check_xlm_robert_large_conll03():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    config = AutoConfig.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    print(config.id2label)

    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, "cejil_staging33.json").read_text())
    # all_text = "\n".join([segment["text"] for segment in segment_boxes])
    classifier = pipeline("ner", model=model, tokenizer=tokenizer)
    # print(classifier("Alya told Jasmine that Andrew could pay with cash.."))
    text = "Alya told she will go to New York"
    result = classifier(text)
    aggregated_entities = aggregate_entities(result)
    print("\n".join([str(entity) for entity in aggregated_entities]))


if __name__ == "__main__":
    # check_bert_base_ner()
    check_xlm_robert_large_conll03()
