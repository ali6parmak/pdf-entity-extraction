from typing import Optional
from ollama_entity_extraction.data_model.EntityInfo import EntityInfo


class EntitiesDict:
    def __init__(self):
        self.entities: dict[str, EntityInfo] = {}

    def add_entity(
        self,
        entity_text: str,
        page_info: str,
        mention: str,
        mention_start: int,
        mention_end: int,
        segment_number: int,
    ):
        if entity_text not in self.entities:
            self.entities[entity_text] = EntityInfo()
        entity_info = self.entities[entity_text]
        entity_info.pages.append(page_info)
        entity_info.mentions.append(mention)
        entity_info.mention_starts.append(mention_start)
        entity_info.mention_ends.append(mention_end)
        entity_info.segment_numbers.append(segment_number)

    def merge_entities(self, target_entity: str, source_entity: str):
        if target_entity not in self.entities:
            self.entities[target_entity] = EntityInfo()
        if source_entity in self.entities:
            self.entities[target_entity].extend(self.entities[source_entity])
            del self.entities[source_entity]

    def to_dict(self) -> dict[str, dict]:
        return {entity_text: entity_info.__dict__ for entity_text, entity_info in self.entities.items()}

    @staticmethod
    def from_dict(data: dict[str, dict]) -> "EntitiesDict":
        entities_dict = EntitiesDict()
        for entity_text, entity_info_data in data.items():
            entity_info = EntityInfo(**entity_info_data)
            entities_dict.entities[entity_text] = entity_info
        return entities_dict

    def keys(self):
        return self.entities.keys()

    def items(self):
        return self.entities.items()

    def get_entity_info(self, entity_text: str) -> Optional[EntityInfo]:
        return self.entities.get(entity_text)

    def pop(self, entity_text: str):
        return self.entities.pop(entity_text, None)

    def sort_entities(self):
        self.entities = dict(sorted(self.entities.items()))
