from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    prompt: str
    answer: str
    code: str
    code_output: str
    examples: str
    explanation: str
    failedTimes: int
    type: str
    additionalResources: list[tuple[str, str]] # list of (url, description) tuples
    