from typing import List
from pydantic import BaseModel, Field

# ========== Pydantic Models for Painpoint Output ==========

class Quote(BaseModel):
    user: str = Field(description="The username of the user who expressed the pain point")
    quote: str = Field(description="A short quote from the text, 1-2 sentences")

class PainPoint(BaseModel):
    title: str = Field(description="The title of the pain point")
    number_of_users: int = Field(description="The number of unique users who expressed the pain point")
    category: str = Field(description="The category of the pain point")
    quote: List[Quote]


class PainPointOutput(BaseModel): 
    pain_points: List[PainPoint]
    number_of_pain_points: int

    class Config:
        extra = "ignore"