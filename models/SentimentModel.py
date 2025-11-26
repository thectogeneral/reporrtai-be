from pydantic import BaseModel, Field
from typing import Optional, List


class SentimentItem(BaseModel):
    user: str = Field(
        description="The username of the user who who wrote the quote"
        )
    sentiment: str = Field(
        description="One of: 'positive', 'negative', or 'neutral'"
    )
    quote: str = Field(
        description="Exact user quote from the thread, or 'Implicit, not explicitly quoted'"
    )
    brief_reason: str = Field(
        description="Short explanation of why this quote fits the sentiment"
    )


class SentimentExtractionOutput(BaseModel):
    sentiments: List[SentimentItem] = Field(
        description="List of extracted sentiment items"
    )
    number_of_sentiments: int = Field(
        description="The number of sentiments extracted from the thread"
    )

    class Config:
        extra = "ignore"