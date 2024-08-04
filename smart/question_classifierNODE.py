from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from .graph_state import GraphState
from .ollama_model import llm

CLASSES = ["code_related", "other", "python"]

promptEN = PromptTemplate(
    template="""
Including example inputs and outputs can indeed help in making the classification clearer and more accurate. Here is the revised prompt with example inputs and outputs:

markdown

You are given a prompt that you will classify.

### Prompt:
{prompt}

### Instructions:
Classify the question into one of the following types:
- `code_related`: This classification applies if the task involves code but does not specifically require Python. It may involve explaining code, writing code in a different language, or writing code for a specific platform or framework. This is the preferred classification if you are unsure whether the task requires Python.
- `other`: This classification applies if the question does not require any code to be answered.
- `python`: This classification applies if the question really needs Python to solve the task, including writing code for logic, calculations, or data manipulation. This classification is specific to Python and should not be used for other programming languages.

### Important:
- Carefully analyze the prompt to determine if it explicitly or implicitly requires Python or any other code to solve the task or answer the question. Look for keywords or phrases that indicate the need for code.
- Provide only the class name as the answer.
- If the task cannot be solved with Python or it specifies a different language, classify it as `code_related` or `other` accordingly.

### Output:
Provide a JSON object with one key `answer` and the classification as a string.

### Example Inputs and Outputs:
#### Example 1:
**Input Prompt:** "Explain how a bubble sort algorithm works." or "Help me write code for Unreal Engine" or "Write a make file for compiling C++ code."
**Classification:** 

{{
"answer": "code_related"
}}


#### Example 2:
**Input Prompt:** "What is the capital of France?" or "Summarize what is on this website. https://www.example.com" or "Summarize this article. https://www.example.com/article"
**Classification:** 

{{
"answer": "other"
}}


#### Example 3:
**Input Prompt:** "Write a Python script to calculate the factorial of a number." or "Calculate 100th fibonacci number."
**Classification:** 

{{
"answer": "python"
}}



Remember there are only three classes: `code_related`, `other`, and `python`. Choose the most appropriate class based on the prompt.

Previous failure or failed attempt:
{failure}
""",
    inputs=["prompt", "failure"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()

def classify_question(state: GraphState) -> str:
    prompt = state["prompt"]
    retries = 0
    answer = {"answer":"other"}
    answerCorrect = False
    failure = "No previous failure"
    while retries < 10 and not answerCorrect:
        try:
            answer = retrieval_classifier.invoke({"prompt": prompt, "failure": failure})
            if "answer" in answer and answer["answer"] in CLASSES:
                answerCorrect = True
                break
            else:
                print("Invalid Classification: {}".format(answer) + " please classify the question into one of the following types: code_related, other, python\n")  
                failure = "Invalid Classification: {} please classify the question into one of the following types: code_related, other, python\n".format(answer)  
        except:
            retries += 1
    
    state["type"] = answer["answer"]
    #print("classified question as: ", state["type"])
    #state["update_process"]("classified question as: " + state["type"] + "\n")
    return state
    
if __name__ == "__main__":
    print(classify_question("I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"))

