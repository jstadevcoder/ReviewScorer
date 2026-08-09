from enum import Enum

class ReviewSeverity(Enum):
    HIGH = ("HIGH", "HIGH severity")
    MEDIUM = ("MEDIUM", "MEDIUM severity")
    LOW = ("LOW", "LOW severity")
    NONE = ("NONE", "NONE severity")

    def __new__(cls, value, description):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.description = description
        return obj
    
    @classmethod
    def list_all_descriptions(cls) -> str:
        lines = [f"{member.name}: {member.description}" for member in cls]
        return "\n".join(lines)
