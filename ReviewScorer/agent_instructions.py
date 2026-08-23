from abstract_agent_instructions import AbstractAgentInstructions
from agent_output import CommentOutput, PRComment, ReviewOutput
from comments_category import CommentsCategory
from review_severity import ReviewSeverity
from pydantic import BaseModel
from typing import Tuple

class ReviewAgentInstructions(AbstractAgentInstructions):
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        return "", None

    def instructions_list(argParams: dict[str, str]) -> list[str]:
        return [f"""
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
                
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        return [f"""
Perform code review on initial diffs only.
Do not do anything on the PR itself, instead return the review in this chat in clear and concise instructions.
Analyze the aggregated diff from step 2. Do not consider or review Any diffs from commits added after PR creation.
Focus on code quality, correctness, style, potential bugs, and adherence to guidelines within these diffs.
""", ReviewOutput]

class ScorerAgentInstructions(AbstractAgentInstructions):
    def starting_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        f"""
For the code reviews performed on the initial diffs. Compare them to all the comments existing in the PR URL {argParams.get("url")}.
From all the comments existing in the PR, do the following:

1) Exclude the comments that the code review already covered in its instuction or reasoning fields
2) Select the comments that have resulted in a code change in the PR or have been helpful to the PR creator in terms of learnings and tips
3) Return the comments that meets all the criteria listed
""", PRComment

    def instructions_list(argParams: dict[str, str]) -> list[str]:
        []
        
    def ending_instruction(argParams: dict[str, str]) -> Tuple[str, type[BaseModel] | None]:
        return f"""
For each of the comments that have met all the criteria previously, score them.
Do not do anything on the PR itself, instead return the scoring with reasoning for the score.
Focus on the comment in regards to the PR code quality, correctness, style, potential bugs, and adherence to guidelines.
Also consider if the comment was helpful to the PR creator in terms of learnings and tips.
""", CommentOutput