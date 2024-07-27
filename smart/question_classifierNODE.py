from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from graph_state import GraphState
from ollama_model import llm

CLASSES = ["needs_code", "other"]

promptEN = PromptTemplate(
    template="""
        You are given a prompt that you will classify.

        ### Prompt:
        {prompt}

        ### Instructions:
        Classify the question into one of the following types:
        - `needs_code`: This classification applies if the question can be solved by writing code. This option is prefered if the prompt requires some logic, calculations, or data manipulation to be done. Generaly math problems should be classified as needs_code.
        - `other`: This classification applies if the question does not require any code to be answered.

        ### Important:
        - Carefully analyze the prompt to determine if it explicitly or implicitly requires code to solve the task or answer the question. Look for keywords or phrases that indicate the need for code.
        - Provide only the class name as the answer.

        ### Output:
        Provide a JSON object with one key `answer` and the classification as a string.

        ### Example JSON Output:

        "answer": "needs_code"
    """,
    inputs=["prompt"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()

def classify_question(state: GraphState) -> str:
    prompt = state["prompt"]
    answer = retrieval_classifier.invoke(prompt)
    state["type"] = answer["answer"]
    #print("classified question as: ", state["type"])
    #state["update_process"]("classified question as: " + state["type"] + "\n")
    return state
    
if __name__ == "__main__":
    print(classify_question("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

