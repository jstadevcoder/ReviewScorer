from pydantic import BaseModel, Field

class PRComment(BaseModel):
    comment: str = Field(..., description="The comment")
    comment_id: str = Field(..., description="The url fragment identifier of the html url for the comment, including the leading #")
    comment_author: str = Field(..., description="The fullname of the person that made the comment")
    pr_link: str = Field(..., description="The url link of the PR the comment is in")