from typing import List
from pydantic import BaseModel, Field

class Topic(BaseModel):
    topic: str = Field(..., description="A concise, descriptive label of the topic discussed")
    ##subtopics: List[str] = Field(default_factory=list, description="List of related subtopics under this topic")
    brief_reason: str = Field(..., description="Short sentence explaining why this topic is relevant in the thread")

class TopicOutput(BaseModel):
    topics: List[Topic] = Field(..., description="List of all relevant topics extracted from the thread")
    number_of_topics: int = Field(..., description="The number of topics extracted from the thread")
