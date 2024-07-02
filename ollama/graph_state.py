from typing import List
from typing_extensions import TypedDict

class GraphState(TypedDict):
    question: str
    answer: str
    code: str
    code_output: str