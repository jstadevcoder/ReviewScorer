from ReviewScorer.agent_instructions.abstract_agent_instructions import AbstractAgentInstructions
from ReviewScorer.agent_outputs.comment_output import CommentOutput
from ReviewScorer.agent_outputs.pr_comment import PRComment
from pydantic import BaseModel
from typing import Tuple

class ScorerAgentInstructions(AbstractAgentInstructions):
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        return [f"""
From all the comments existing in the PR URL {argParams.get("url")}

1) Exclude the comments listed here:
{argParams.get("comments")}
2) Select the comments that have resulted in a code change in the PR or have been helpful to the PR creator in terms of learnings and tips
3) Return the comments that meets all the criteria listed
""", PRComment]

    def instructions_list(argParams: dict[str, str]) -> list[str]:
        []
        
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        return f"""
For each of the comments that have met all the criteria previously, score them from 1-5 with 5 being very critical or useful and 1 being less impactful.
Do not do anything on the PR itself, instead return the scoring with reasoning for the score.
Focus on the comment in regards to the PR code quality, correctness, style, potential bugs, and adherence to guidelines.
Also consider if the comment was helpful to the PR creator in terms of learnings and tips.
""", CommentOutput