from ReviewScorer.agent_outputs.pr_comment import PRComment
from ReviewScorer.enums.comments_category import CommentsCategory
from ReviewScorer.enums.review_severity import ReviewSeverity
from pydantic import Field

class CommentOutput(PRComment):
    reason: str = Field(..., description="The reason for the comment")
    comment_category: CommentsCategory = Field(..., description=
        f"""The category the instruction belongs to.
        This is an enum with the following values:
        {CommentsCategory.list_all_descriptions()}""")
    severity: ReviewSeverity = Field(..., description=
        f"""The serverity of the instruction.
        This is an enum with the following values:
        {ReviewSeverity.list_all_descriptions()}""")
    score: int = Field(..., description="The score for the comment")