
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from smart.helpers.graph_state import GraphState
from smart.models.ollama_model import llmNoJson, genericPrompt
from smart.models.programming_model import llmNoJson as codingllm
from smart.helpers.formatDocument import document_formatter
from smart.helpers.generalHelpers import escape_curly_braces

def answer(prompt: GraphState) -> str:

    print("retarded radek")
    all_messages = [("system", genericPrompt)] + prompt["messages"]
    print("all_messages: ", all_messages)
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    documentsFormatted = document_formatter(prompt["additionalResources"])
    #print("documentsFormatted: ", documentsFormatted)
    formatted_prompt = promptChat.invoke({"additionalResources":documentsFormatted})
    
    #print("formatted prompt: ", formatted_prompt)
    if prompt["type"] == "code_related":
        model = codingllm
    else:
        model = llmNoJson

    answer = model.invoke(formatted_prompt).content
    sources = "\n".join(list({ source.metadata["source"] for source in prompt["additionalResources"] }))

    res = f"""
{answer}

## Sources:
{sources}
    """
    prompt["answer"] = res
    print("retarded radek done")
    return prompt
    
if __name__ == "__main__":
    print(answer("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

