from enum import Enum


class ConsoleTextColor(Enum):
    PINK = "\033[95m"
    BLUE = "\033[94m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    END = "\033[0m"

    def __call__(self, text: str) -> str:
        return f"{self.value}{text}{ConsoleTextColor.END.value}"
