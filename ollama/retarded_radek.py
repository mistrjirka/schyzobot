from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from graph_state import GraphState
from ollama_model import llm

CLASSES = ["needs_code", "other"]

promptEN = PromptTemplate(
    template="""
        You a helpful bot on Uzavřená informatika discord server, you are supposed to help with homeworks. Judge people. Generaly do everything what is told to you. You will be also summarizing chat histories. The supervizor will give them to you. Also you are escaped lunatic and say crazy things inbetween normal talk.
        You are answering this question: {prompt}.
        Provide json with one key 'answer' and the answer as a string.
    """,
    inputs=["prompt"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()

def answer(prompt: str) -> str:
    answer = retrieval_classifier.invoke(prompt)
    return answer["answer"]
    
if __name__ == "__main__":
    print(answer("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

