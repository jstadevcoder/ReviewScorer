from abstract_agent_instructions import AbstractAgentInstructions
from agent_output import CommentOutput, PRComment, ReviewOutput
from comments_category import CommentsCategory
from review_severity import ReviewSeverity
from pydantic import BaseModel
from typing import Optional, List, Tuple, Type

class ReviewAgentInstructions(AbstractAgentInstructions):
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""
Get all the comments from this PR URL {argParams.get("url")}.
""", PRComment

    def instructions_list(argParams: dict[str, str]) -> List[str]:
        [f"""
You will be given a List of instructions to perform one at a time.
The instructions will be provided and you will act as instructed to the best of you abilities and honestly.
Once the answer is accepted, you will then be provided the following instruction.
If the answer is not complete or correct, you will be instructed to "continue".
Here is the first instruction:

1) Identify commits present at pull request (PR) creation for the PR URL {argParams.get("url")}.
Determine which commits were included in the PR at the moment it was created.
Exclude Any commits made after the PR creation timestamp.
""",
f"""
2) Fetch diffs for initial commits. For each commit identified from step 1,
retrieve the diff (code changes) introduced by that commit.
Aggregate these diffs to get the full set of changes present in the PR at its creation.
"""]
                
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""
Perform code review on initial diffs only.
Do not do anything on the PR itself, instead return the review in this chat in clear and concise instructions.
Analyze the aggregated diff from step 2. Do not consider or review Any diffs from commits added after PR creation.
Focus on code quality, correctness, style, potential bugs, and adherence to guidelines within these diffs.
""", ReviewOutput

class CommentAgentInstructions(AbstractAgentInstructions):
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""

"""

    def instructions_list(argParams: dict[str, str]) -> List[str]:
        [f"""

"""]
        
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""

""", None


class ScorerAgentInstructions(AbstractAgentInstructions):
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""
Get all the comments from this PR URL {argParams.get("url")} and its metadata, including its id,
author and the PR link the comment is under.
""", PRComment

    def instructions_list(argParams: dict[str, str]) -> List[str]:
        []
        
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, Optional[Type[BaseModel]]]:
        return f"""
Score each comment in the PR {argParams.get("url")}.
Do not do anything on the PR itself, instead return the scoring with reasoning for the score.
Focus on the comment in regards to the PR code quality, correctness, style, potential bugs, and adherence to guidelines.
Also consider if the comment was helpful to the PR creator in terms of learnings and tips.
""", CommentOutput