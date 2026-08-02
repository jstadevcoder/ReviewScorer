from abc import ABC, abstractmethod
from typing import Tuple

class AbstractAgentInstructions(ABC):
    @abstractmethod
    def review_starting_instruction() -> str:
        pass

    @abstractmethod
    def review_instructions_list(argParams: dict[str, str]) -> list[str]:
        pass

    @abstractmethod
    def review_ending_instruction(argParams: dict[str, str]) -> Tuple[str, type]:
        pass

    @abstractmethod
    def comment_starting_instruction() -> str:
        pass

    @abstractmethod
    def comment_instructions_list(argParams: dict[str, str]) -> list[str]:
        pass

    @abstractmethod
    def comment_ending_instruction(argParams: dict[str, str]) -> Tuple[str, type]:
        pass

    @abstractmethod
    def scorer_starting_instruction() -> str:
        pass

    @abstractmethod
    def scorer_instructions_list(argParams: dict[str, str]) -> list[str]:
        pass

    @abstractmethod
    def scorer_ending_instruction(argParams: dict[str, str]) -> Tuple[str, type]:
        pass