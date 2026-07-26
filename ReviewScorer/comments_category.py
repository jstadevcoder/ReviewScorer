from enum import Enum

class CommentsCategory(Enum):
    REFACTOR = ("Refactor", "The comment is related to refactor work")
    SECURITY = ("Security", "The comment is related security issues")
    ERROR = ("Error", "The comment is related to error handling")
    NOP = ("NOP", "The comment is just a comment that does nothing")

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj

    def list_all_descriptions(cls) -> str:
        lines = [f"{member.name} ({member.value}): {member.description}" for member in cls]
        return "\n".join(lines)
