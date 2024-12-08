from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from smart.helpers.graph_state import GraphState, STATUS_BLOCK_END
from smart.models.ollama_model import llmChainOfThought,llmNoJson, genericPrompt
from smart.models.programming_model import llmNoJson as codingllm
from smart.helpers.formatDocument import document_formatter
from smart.helpers.generalHelpers import escape_curly_braces

def answer(prompt: GraphState) -> str:

    all_messages = [("system", genericPrompt)] + prompt["messages"]
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    documentsFormatted = document_formatter(prompt["additionalResources"])
    
    if "update_process" in prompt:
        prompt["update_process"](documentsFormatted)
    if prompt["type"] == "code_related":
        model = llmNoJson#llmChainOfThought
        if "update_process" in prompt:
            prompt["update_process"]("Question is code related\n")
    else:
        model = llmNoJson#llmChainOfThought
        if "update_process" in prompt:
            prompt["update_process"]("Question is not code related\n")

    
    if "update_process" in prompt:
        prompt["update_process"](STATUS_BLOCK_END)  # End status block before starting LLM
    formatted_prompt = promptChat.invoke({"additionalResources":documentsFormatted})
    print(formatted_prompt)
    # Stream the actual LLM response
    answer_content = []
    for chunk in model.stream(formatted_prompt):
        if "update_process" in prompt:
            prompt["update_process"](chunk.content)
            answer_content.append(chunk.content)

    sources = "\n".join(list({ source.metadata["source"] for source in prompt["additionalResources"] }))

    if "update_process" in prompt:
        prompt["update_process"]("## References:\n" + sources)

    res = f"""
{''.join(answer_content)}

## References:
{sources}
    """
    prompt["answer"] = res
    return prompt
    
if __name__ == "__main__":
    print(answer("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

