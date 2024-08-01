
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from .graph_state import GraphState
from .ollama_model import llmNoJson, genericPrompt
from .programming_model import llmNoJson as codingllm
CLASSES = ["needs_code", "other"]



def answer(prompt: GraphState) -> str:

    print("retarded radek")
    all_messages = [("system", genericPrompt)] + prompt["messages"]
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    documentsBySource = dict()
    
    for source in prompt["additionalResources"]:
        if source.metadata["source"] not in documentsBySource:
            documentsBySource[source.metadata["source"]] = ""
        documentsBySource[source.metadata["source"]]+= source.page_content + "\n"    
    
    documentsFormatted = "\n".join([f"Content of {source}: {documentsBySource[source]}" for source in documentsBySource])
    formatted_prompt = promptChat.format(additionalResources=documentsFormatted)
    
    print("formatted prompt: ", formatted_prompt)
    answer = llmNoJson.invoke(formatted_prompt).content
    sources = "\n".join(list({ source.metadata["source"] for source in prompt["additionalResources"] }))

    res = f"""
{answer}

## Sources:
{sources}
    """
    prompt["answer"] = res
    
    return prompt
    
if __name__ == "__main__":
    print(answer("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

