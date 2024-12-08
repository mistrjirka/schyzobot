from langchain_ollama import ChatOllama
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel

class BinaryResponse(BaseModel):
    result: bool

# Create JSON parser for binary responses - allowing additional fields
binary_parser = JsonOutputParser(
    pydantic_schema={
        "type": "object",
        "properties": {
            "result": {"type": "boolean"},
            # Allow any additional properties
        },
        "required": ["result"],
        "additionalProperties": True  # This makes it accept additional fields
    }
)

# select ollama model
MODEL = "llama3.3:70b-instruct-q2_K" #"llama3.1:8b-instruct-q8_0" #"dolphin-llama3:8b-v2.9-q8_0"#"dolphin-llama3:70b"#"dolphin-llama3:8b-v2.9-fp16"
MODEL_SMALL = "llama3.1:8b-instruct-q8_0"
MODEL_CHAIN_OF_THOUGHT = "qwq:32b"
# higher temparature means higher creativity
TEMPERATURE = 0.45
CREATIVE_TEMPERATURE = 0.6
print("Loading ollama model...", end=" ", flush=True)
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, format="json")

llmSmall = ChatOllama(model=MODEL_SMALL, temperature=TEMPERATURE, format="json")

llmSmallNoJson = ChatOllama(model=MODEL_SMALL, temperature=TEMPERATURE)

# Replace binary models with JSON-enabled versions
llmSmallBinary = ChatOllama(model=MODEL_SMALL, temperature=TEMPERATURE, format="json")
llmBinary = ChatOllama(model=MODEL, temperature=TEMPERATURE, format="json")

llmNoJson = ChatOllama(model=MODEL, temperature=TEMPERATURE)

llmNoJsonCreative = ChatOllama(model=MODEL, temperature=CREATIVE_TEMPERATURE)
llmChainOfThought = ChatOllama(model=MODEL_CHAIN_OF_THOUGHT)


print("DONE")

genericPrompt = """
You are a researcher who must strictly adhere to the facts provided in the Context section. Do not include any information outside of this context.
Just answer the question based on the information provided in the Context section. No additional information is needed.
Context:
{additionalResources}
"""
