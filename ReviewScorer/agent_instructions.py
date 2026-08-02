from abstract_agent_instructions import AbstractAgentInstructions
from agent_output import ReviewOutput
from comments_category import CommentsCategory
from review_severity import ReviewSeverity
from pydantic import BaseModel
from typing import Optional, Tuple, Type

class AgentInstructions(AbstractAgentInstructions):
    def review_starting_instruction() -> str:
        return f"""
You will be given a list of instructions to perform one at a time.
The instructions will be provided and you will act as instructed to the best of you abilities and honestly.
Once the answer is accepted, you will then be provided the following instruction.
If the answer is not complete or correct, you will be instructed to "continue".
Here is the first instruction:
"""

    def review_instructions_list(argParams: dict[str, str]) -> list[str]:
        [f"""
1) Identify commits present at pull request (PR) creation for the PR URL {argParams.get["url"]}.
Determine which commits were included in the PR at the moment it was created.
Exclude any commits made after the PR creation timestamp.
""",
f"""
2) Fetch diffs for initial commits. For each commit identified from step 1,
retrieve the diff (code changes) introduced by that commit.
Aggregate these diffs to get the full set of changes present in the PR at its creation.
""",
f"""
Perform code review on initial diffs only.
Do not do anything on the PR itself, instead return the review in this chat in clear and concise instructions.
Analyze the aggregated diff from step 2. Do not consider or review any diffs from commits added after PR creation.
Focus on code quality, correctness, style, potential bugs, and adherence to guidelines within these diffs.
"""]
                
    def review_ending_instruction() -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""
""", ReviewOutput
    
    def comment_starting_instruction() -> str:
        return f"""

"""

    def comment_instructions_list(argParams: dict[str, str]) -> list[str]:
        [f"""

"""]
        
    def comment_ending_instruction() -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""

""", None

    def scorer_starting_instruction() -> str:
        return f"""

"""

    def scorer_instructions_list(argParams: dict[str, str]) -> list[str]:
        [f"""

"""]
        
    def comment_ending_instruction() -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""

""", None