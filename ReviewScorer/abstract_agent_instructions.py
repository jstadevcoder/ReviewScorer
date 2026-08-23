from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Tuple

class AbstractAgentInstructions(ABC):
    @abstractmethod
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        pass

    @abstractmethod
    def instructions_list(argParams: dict[str, str]) -> list[str]:
        pass

    @abstractmethod
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        pass