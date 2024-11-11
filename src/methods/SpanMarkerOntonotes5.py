from pdf_features.Rectangle import Rectangle
from span_marker import SpanMarkerModel

from data_model.EntityBox import EntityBox
from data_model.WordBox import WordBox
from methods.NERTransformerModel import NERTransformerModel


class SpanMarkerOntonotes5(NERTransformerModel):
    def __init__(self, model_name: str, show_logs: bool = False, initialize_auto_model: bool = True):
        super().__init__(model_name, show_logs, initialize_auto_model)
        self.classifier = SpanMarkerModel.from_pretrained(model_name)
        self.classifier.cuda()

    def process_segment(self, pdf_words, segment_box, total_entity_count) -> list[EntityBox]:
        segment_bounding_box = Rectangle.from_width_height(
            segment_box["left"], segment_box["top"], segment_box["width"], segment_box["height"]
        )
        word_boxes_for_page = pdf_words.pdf_words[segment_box["page_number"] - 1]
        word_boxes_in_segment: list[WordBox] = WordBox.find_word_boxes_in_rectangle(
            segment_bounding_box, word_boxes_for_page
        )
        result = self.classifier.predict(" ".join([wb.text for wb in word_boxes_in_segment]))
        aggregated_entities = []
        for r in result:
            aggregated_entities.append(
                {
                    "text": r["span"],
                    "entity_label": r["label"],
                    "start_index": r["char_start_index"],
                    "end_index": r["char_end_index"],
                }
            )

        print(result)

        if segment_box["page_number"] == 1:
            print("\n".join([str(r) for r in aggregated_entities]))

        total_entity_count += len(aggregated_entities)
        return self.create_entity_boxes(aggregated_entities, segment_box, word_boxes_in_segment)


if __name__ == "__main__":
    model_name = "tomaarsen/span-marker-roberta-large-ontonotes5"
    show_logs = False
    initialize_auto_model = False
    span_marker_ontonotes5 = SpanMarkerOntonotes5(model_name, show_logs, initialize_auto_model)
    span_marker_ontonotes5.get_entities("cejil_staging33", save_output=True)
