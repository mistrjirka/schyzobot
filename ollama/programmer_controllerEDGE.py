from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from graph_state import GraphState
from ollama_model import llm



promptEN = PromptTemplate(
    template="""
        **Task: Verify the correctness of a program's execution**

        You are provided with the following:

        - The initial prompt describing the task to be accomplished.
        - The code to be evaluated.
        - The output produced by running the code (Examples Run).
        - An explanation of the code.
        - Example code snippets demonstrating how to use the provided code.

        Your task is to determine if the code execution and its result meet the requirements of the initial prompt. Use the explanation as a reference but prioritize the correctness of the code and its output. If the code is correct but the explanation is incorrect, still judge the code as correct.

        ### Inputs:
        - **Initial Prompt**: {prompt}
        - **Code**: {code}
        - **Examples Run**: {examples_run}
        - **Code Explanation**: {explanation}
        - **Examples**: {examples}

        ### Instructions:
        1. Verify that the code runs without errors.
        2. Check if the output of the code matches the expected results as per the initial prompt.
        3. Use the provided explanation for additional context but do not let inaccuracies in the explanation affect the judgment of the code's correctness.
        4. If there are any expected outputs mentioned in comments within the code, ensure they match the actual output! The example code will maybe contain # output: ... comments. Be sure to check if the output matches the expected output.
        5. If the code contains errors or the output does not meet the prompt requirements, consider it incorrect.

        ### Output:
        Provide a JSON object with the key `answer` and a boolean value:
        - `true` if the code is correct and meets the prompt requirements.
        - `false` if the code is incorrect or does not meet the prompt requirements.
    """,
    inputs=["prompt", "code_output", "explanation", "prompt", "examples"]
)
retrieval_classifier = promptEN | llm | JsonOutputParser()

def judgeProgram(program: GraphState) -> bool:
    print("judging program")
    code_execution_result = program["code_output"]
    print(code_execution_result)
    print(program["code"])
    valid = False
    retries = 0
    answer = None
    while not valid and retries < 5:
        answer = retrieval_classifier.invoke(
            {
                "code": program["code"], 
                "examples_run": code_execution_result, 
                "explanation": program["explanation"], 
                "prompt": program["prompt"],
                "examples": program["examples"]
            }
        )
        if("answer" in answer and answer["answer"] in [True, False]):
            valid = True
        else:
            retries += 1

    return answer["answer"]
    
if __name__ == "__main__":
    import programmer_modelNODE
    prompt = {}
    prompt["prompt"] = "I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"
    prompt["previous_result"] = "None"
    prompt["previous_code"] = "None"
    result = programmer_modelNODE.makeProgram(prompt)

    print(judgeProgram(result))

