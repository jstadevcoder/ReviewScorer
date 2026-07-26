from comments_category import CommentsCategory
from pydantic import BaseModel, Field

class ReviewOutput(BaseModel):
    instruction: str = Field(description="The instruction of the review")
    instruction_category: CommentsCategory = Field(description=
        f"""The category the instruction belongs to.
        This is an enum with the following values:
        {CommentsCategory}""")