from gliner import GLiNER
from pdf_features.Rectangle import Rectangle
from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel

def print_with_line_breaks(text, line_length=150):
    for i in range(0, len(text), line_length):
        print(text[i:i+line_length])

class GLiNERModel(NERTransformerModel):
    def __init__(self, model_name: str, show_logs: bool = False):
        super().__init__(model_name, show_logs, initialize_auto_model=False)
        self.model_name = model_name
        self.classifier = GLiNER.from_pretrained(model_name)
        self.show_logs = show_logs

    def process_segment(self, pdf_words, segment_box, total_entity_count) -> list[EntityBox]:
        segment_bounding_box = Rectangle.from_width_height(
            segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        )
        word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
        word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(
            segment_bounding_box, word_boxes_for_page
        )

        labels = ["date"]
        entities = []
        full_text = " ".join([wb.text for wb in word_boxes_in_segment])
        previous_chunk_index_end = 0
        chunk_text = ""
        for word_box in word_boxes_in_segment:
            if len(chunk_text) + len(word_box.text) < 375:
                chunk_text += word_box.text + " "
                continue
            chunk_text += word_box.text

            chunk_entities = self.classifier.predict_entities(chunk_text, labels)

            for entity in chunk_entities:
                entity["start"] += previous_chunk_index_end
                entity["end"] += previous_chunk_index_end

            previous_chunk_index_end += len(chunk_text) + 1

            entities.extend(chunk_entities)
            chunk_text = ""

        if chunk_text:
            chunk_entities = self.classifier.predict_entities(chunk_text, labels)

            for entity in chunk_entities:
                entity["start"] += previous_chunk_index_end
                entity["end"] += previous_chunk_index_end

            previous_chunk_index_end += len(chunk_text) + 1

            entities.extend(chunk_entities)


        aggregated_entities = [
            {
                "text": entity["text"],
                "entity_label": entity["label"][: 3 if len(entity["label"]) > 4 else len(entity["label"])].upper(),
                "start_index": entity["start"],
                "end_index": entity["end"],
            }
            for entity in entities
        ]

        print("PAGE: ", segment_box["page_number"])
        print_with_line_breaks(" ".join([wb.text for wb in word_boxes_in_segment]))
        print("-" * 30)
        print("\n".join([str(r) for r in aggregated_entities]))
        print("*" * 30)

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)
