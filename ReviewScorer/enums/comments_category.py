from enum import Enum

class CommentsCategory(Enum):
    REFACTOR = ("REFACTOR", "The comment is related to REFACTOR work")
    SECURITY = ("SECURITY", "The comment is related SECURITY issues")
    PERFORMANCE = ("PERFORMANCE", "The comment is related PERFORMANCE")
    ERROR = ("ERROR", "The comment is related to ERROR handling")
    INFORMATIVE = ("INFORMATIVE", "The comment is providing extra knowledge")
    NOP = ("NOP", "The comment is just a comment that does nothing")
    OTHER = ("OTHER", "The comment is related to something else")

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj
    
    @classmethod
    def list_all_descriptions(cls) -> str:
        lines = [f"{member.name}: {member.description}" for member in cls]
        return "\n".join(lines)
