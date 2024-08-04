import re
from smart.helpers.graph_state import GraphState

def extract_links_from_prompt(prompt_text):
    # Use regex to find all links in the prompt text
    links = re.findall(r'(https?://\S+)', prompt_text)
    return links


def classify_prompt_researchNeeded(prompt: GraphState) -> str:
    
    if len(prompt["links"]) > 0:
        return "SUMMARIZE"
    else:
        return "RESEARCH"
    