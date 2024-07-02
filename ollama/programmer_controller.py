from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import subprocess
import sys
from graph_state import GraphState
from ollama_model import llm

#to change
def execute_python_code(code):
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout + result.stderr
    except Exception as e:
        output = str(e)
    return output

promptEN = PromptTemplate(
    template="""
        You are given a code, result of the code after it is runned, explanation of the code and initial prompt. The initial prompt contains task that needs to be accomplished you judge if the code execution result and code are correct. You can judge if the code is correct with help of the explanation, but if the code is correct and explanation is wrong you should still judge the code as correct.
        
        Here is the code: {code}

        Here is the result of the code: {result}
        
        Here is the explanation of the code: {explanation}
        
        You have to classify the question as one of the following types: is correct, is incorrect. Where is_correct means that the code is correct and is_incorrect means that the code is incorrect. 
        Provide JSON with one key 'answer' and the answer as a boolean. So if the code is correct the answer should be true and if the code is incorrect the answer should be false.
    """,
    inputs=["prompt", "previous_result"]
)
retrieval_classifier = promptEN | llm | JsonOutputParser()

def judgeProgram(program: dict) -> str:
    code_execution_result = execute_python_code(program["code"])
    print(code_execution_result)
    answer = retrieval_classifier.invoke({"code": program["code"], "result": code_execution_result, "explanation": program["explanation"]})
    return answer["answer"]
    
if __name__ == "__main__":
    import programmer_model
    prompt = {}
    prompt["prompt"] = "I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"
    prompt["previous_result"] = "None"
    prompt["previous_code"] = "None"
    result = programmer_model.makeProgram(prompt)

    print(judgeProgram(result))

