from typing import List, Callable
from typing_extensions import TypedDict

# Using markdown headers and separators for clear visual separation
STATUS_BLOCK_START = """
<details>
<summary>Status Progress</summary>

### Status Progress
---
"""

STATUS_BLOCK_END = """
---
</details>

# Response
"""

class GraphState(TypedDict):
    prompt: str
    answer: str
    code: str
    code_output: str
    examples: str
    explanation: str
    failedTimes: int
    type: str
    additionalResources: list[tuple[str, str]]  # list of (url, description) tuples
    links: list[str]
    messages: list[tuple[str,str]]
    update_process: Callable[[str], None]  # Function that takes a string and returns None

    @staticmethod
    def is_status_message(message: str) -> bool:
        """Check if a message is a status update"""
        status_emojis = ["🔍", "🌐", "📚", "✅", "💭", "🤔", "✨", "🚀"]
        return any(emoji in message for emoji in status_emojis)

    @staticmethod
    def is_status_block(message: str) -> bool:
        """Check if a message is within the status block"""
        return message.startswith(STATUS_BLOCK_START) or STATUS_BLOCK_END in message
