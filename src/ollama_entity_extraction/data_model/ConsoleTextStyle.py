from enum import Enum

class ConsoleTextStyle(Enum):
    BOLD = "\033[1m"
    ITALIC = "\033[3m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"

    @staticmethod
    def apply(text: str, styles: list['ConsoleTextStyle']) -> str:
        style_codes = ''.join(style.value for style in styles)
        return f"{style_codes}{text}{ConsoleTextStyle.END.value}"

    def __call__(self, text: str) -> str:
        return f"{self.value}{text}{ConsoleTextStyle.END.value}"
