from comments_category import CommentsCategory
from pydantic import BaseModel, Field
from review_severity import ReviewSeverity

class ReviewOutput(BaseModel):
    instruction: str = Field(..., description="The instruction of the review")
    reason: str = Field(..., description="The reason for the review instruction")
    instruction_category: CommentsCategory = Field(..., description=
        f"""The category the instruction belongs to.
        This is an enum with the following values:
        {CommentsCategory.list_all_descriptions()}""")
    severity: ReviewSeverity = Field(..., description=
        f"""The serverity of the instruction.
        This is an enum with the following values:
        {ReviewSeverity.list_all_descriptions()}""")

class PRComment(BaseModel):
    comment: str = Field(..., description="The comment")
    comment_id: str = Field(..., description="The guid of the comment")
    comment_author: str = Field(..., description="The fullname of the person that made the comment")
    pr_link: str = Field(..., description="The url link of the PR the comment is in")

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