import re
from smart.helpers.graph_state import GraphState

def extract_links_from_prompt(prompt_text):
    # Use regex to find all links in the prompt text
    links = re.findall(r'(https?://\S+)', prompt_text)
    return links

def links_prompt(prompt: GraphState) -> GraphState:
    print("linksPrompt")
    extracted_links = extract_links_from_prompt(prompt["prompt"])
    prompt["links"] = extracted_links
    prompt["additionalResources"] = []
    return prompt
