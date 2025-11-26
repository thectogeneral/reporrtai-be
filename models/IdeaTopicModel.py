from pydantic import BaseModel, Field
from typing import List

class ProductIdea(BaseModel):
    product_name: str = Field(
        description="The name of the product idea."
    )
    tagline: str = Field(
        description="A short, one-sentence tagline summarizing the idea."
    )
    description: str = Field(
        description="A 2–4 sentence explanation of the product and how it works."
    )
    pain_points_solved: List[str] = Field(
        description="List of pain point titles or summaries that this product idea directly addresses."
    )

class IdeaTopicOutput(BaseModel):
    product_ideas: List[ProductIdea] = Field(
        description="List of product ideas generated from the conversation and pain points."
    )
