import json
from pathlib import Path

from flair.data import Sentence
from flair.nn import Classifier

from configuration import CACHED_DATA_PATH, REFERENCE_DATA_PATH


def create_reference_data(file_name: str):
    classifier = Classifier.load("ner-ontonotes-large")
    segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{file_name}.json").read_text())
    data_string = ""
    data = []
    all_labels = []
    for segment_box in segment_boxes:
        segment_text = " ".join([word for word in segment_box["text"].split()])
        sentence = Sentence(segment_text)
        classifier.predict(sentence)
        entities = sentence.get_spans("ner")
        # cardinals_and_laws = " ".join([entity.text for entity in entities if entity.tag in {"CARDINAL", "LAW"}])
        # cardinals_and_laws = [entity.text for entity in entities if entity.tag in {"CARDINAL", "LAW"}]
        cardinals = [entity.text for entity in entities if entity.tag == "CARDINAL"]
        laws = [entity.text for entity in entities if entity.tag == "LAW"]
        context = " ".join([entity.text for entity in entities if entity.tag in {"CARDINAL", "LAW"}])
        all_labels += [entity.tag for entity in entities]
        # if not cardinals_and_laws:
        #     continue
        if not laws:
            continue
        data.append({"cardinals": cardinals, "laws": laws, "context": context})
        # data_string += cardinals_and_laws + "\n"

    # Path(REFERENCE_DATA_PATH, f"{file_name}_reference_data.txt").write_text(data_string)
    Path(REFERENCE_DATA_PATH, f"{file_name}_reference_data.json").write_text(json.dumps(data))
    print(list(set(all_labels)))


if __name__ == "__main__":
    create_reference_data("cejil_staging33")
