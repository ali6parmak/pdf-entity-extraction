import sys
from pathlib import Path

# Add src directory to path
src_dir = Path(__file__).parent.parent
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

import json
from pdf_features.Rectangle import Rectangle
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
from data_model.EntityBox import EntityBox
from data_model.PdfWords import PdfWords
from data_model.WordBox import WordBox
from save_visualization_to_pdf import save_output_to_pdf
from configuration import CACHED_DATA_PATH, PDFS_PATH


class LegalBertBase:
    def __init__(self, model_name: str, show_logs: bool = False, initialize_auto_model: bool = True):
        self.model_name = model_name
        self.show_logs = show_logs
        self.classifier = None
        if initialize_auto_model:
            self.initialize_auto_model()

    def initialize_auto_model(self):
        tokenizer = AutoTokenizer.from_pretrained("nlpaueb/legal-bert-base-uncased")
        model = AutoModelForTokenClassification.from_pretrained("nlpaueb/legal-bert-base-uncased")
        self.classifier = pipeline("ner", model=model, tokenizer=tokenizer)

    @staticmethod
    def aggregate_entities(ner_results: list[dict]):
        entities = []
        current_entity = None

        for entity_dict in ner_results:
            entity_label = entity_dict["entity"]
            if "-" in entity_label:
                dash_index = entity_label.index("-")
                entity_label = entity_label[dash_index + 1 :]
            text = entity_dict["word"].replace("▁", " ").replace("#", "").strip()
            # Because some entities might only include the characters above
            if not text:
                continue

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

    def get_entities(self, file_name: str, save_output: bool = False):
        pdf_words: PdfWords = PdfWords.from_pdf_path(PDFS_PATH / f"{file_name}.pdf")
        segment_boxes: list[dict] = json.loads((CACHED_DATA_PATH / f"{file_name}.json").read_text())
        entity_boxes: list[EntityBox] = self.process_segments(pdf_words, segment_boxes)
        if self.show_logs:
            print("ENTITY BOX COUNT: ", len(entity_boxes))
        if save_output:
            self.save_visualizations(file_name, entity_boxes, segment_boxes)

    def save_visualizations(self, file_name: str, entity_boxes: list[EntityBox], segment_boxes: list[dict]):
        output_file_name = file_name.replace(".pdf", "") + f"_{self.model_name.replace('/', '_')}.pdf"
        save_output_to_pdf(PDFS_PATH / f"{file_name}.pdf", segment_boxes, output_file_name, entity_boxes)

    def process_segments(self, pdf_words, segment_boxes) -> list[EntityBox]:
        entity_boxes: list[EntityBox] = []
        total_entity_count = 0
        for segment_box in segment_boxes:
            segment_entities = self.process_segment(pdf_words, segment_box, total_entity_count)
            entity_boxes.extend(segment_entities)
        return entity_boxes

    def process_segment(self, pdf_words, segment_box, total_entity_count) -> list[EntityBox]:
        segment_bounding_box = Rectangle.from_width_height(
            segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        )
        word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
        word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(
            segment_bounding_box, word_boxes_for_page
        )
        result = self.classifier(" ".join([wb.text for wb in word_boxes_in_segment]))
        aggregated_entities = LegalBertBase.aggregate_entities(result)

        if segment_box["page_number"] == 1:
            print("\n".join([str(r) for r in aggregated_entities]))

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)

    def create_entity_boxes(self, aggregated_entities, segment_box, word_boxes_in_segment):
        entity_boxes: list[EntityBox] = []
        for entity in aggregated_entities:
            word_boxes_for_entity = WordBox.find_word_boxes_from_indices(
                word_boxes_in_segment, entity["start_index"], entity["end_index"]
            )
            if not word_boxes_for_entity and self.show_logs:
                print(
                    "NO WORD BOXES FOUND FOR ENTITY: ",
                    entity["text"],
                    entity["entity_label"],
                    segment_box["page_number"],
                    entity["start_index"],
                    entity["end_index"],
                )
                continue
            if not word_boxes_for_entity:
                continue
            entity_boxes.append(EntityBox.from_word_boxes(word_boxes_for_entity, entity["entity_label"], entity["text"]))

        return entity_boxes


if __name__ == "__main__":
    legal_bert_base = LegalBertBase("nlpaueb_legal-bert-base-uncased")
    legal_bert_base.get_entities("cejil_staging33", save_output=True)
