from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import List, Optional, Tuple, Type

class AbstractAgentInstructions(ABC):
    @abstractmethod
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        pass

    @abstractmethod
    def instructions_list(argParams: dict[str, str]) -> List[str]:
        pass

    @abstractmethod
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        pass