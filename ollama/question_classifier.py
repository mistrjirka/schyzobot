from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from graph_state import GraphState
from ollama_model import llm

CLASSES = ["needs_code", "other"]

promptEN = PromptTemplate(
    template="""
        You are given a prompt, that you will classify. 
        {prompt}        
        You have to classify the question as one of the following types: needs_code, other. Where needs code means that the question requires to program code to be answered or it requests code. Other means that the question does not require code to be answered.
        Give only the class name as the answer.
        Provide json with one key 'answer' and the answer as a string.
    """,
    inputs=["prompt"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()

def classify_question(prompt: str) -> str:
    answer = retrieval_classifier.invoke(prompt)
    return answer["answer"]
    
if __name__ == "__main__":
    print(classify_question("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

