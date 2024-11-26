from dataclasses import dataclass, field


@dataclass
class EntityInfo:
    pages: list[str] = field(default_factory=list)
    mentions: list[str] = field(default_factory=list)
    mention_starts: list[int] = field(default_factory=list)
    mention_ends: list[int] = field(default_factory=list)
    segment_numbers: list[int] = field(default_factory=list)

    def extend(self, other: "EntityInfo"):
        self.pages.extend(other.pages)
        self.mentions.extend(other.mentions)
        self.mention_starts.extend(other.mention_starts)
        self.mention_ends.extend(other.mention_ends)
        self.segment_numbers.extend(other.segment_numbers)
