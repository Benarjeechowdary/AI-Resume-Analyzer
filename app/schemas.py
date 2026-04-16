from pydantic import BaseModel,Field
from typing import List

class Resume(BaseModel):
    score: int = Field(ge=0, le=95)
    strengths: List[str]
    weaknesses: List[str]
    missing: List[str]
    suggestions: List[str]