import json
from pathlib import Path

from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

from configuration import CACHED_DATA_PATH


def check_bert_base_ner():
    tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
    model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, "cejil_staging33.json").read_text())
    all_text = "\n".join([segment["text"] for segment in segment_boxes])
    nlp = pipeline("ner", model=model, tokenizer=tokenizer)
    # example = "My name is Wolfgang and I live in New York"
    ner_results = nlp(all_text)
    print(ner_results)


def check_xlm_robert_large_conll03():
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-large-finetuned-conll03-english")
    segment_boxes: list[dict] = json.loads(Path(CACHED_DATA_PATH, "cejil_staging33.json").read_text())
    all_text = "\n".join([segment["text"] for segment in segment_boxes])
    classifier = pipeline("ner", model=model, tokenizer=tokenizer)
    # classifier("Alya told Jasmine that Andrew could pay with cash..")
    print(classifier(all_text))


if __name__ == "__main__":
    # check_bert_base_ner()
    check_xlm_robert_large_conll03()
