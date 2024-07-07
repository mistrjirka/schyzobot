from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import subprocess
import sys
from graph_state import GraphState
from ollama_model import llm
import re
max_retries = 50

promptEN = PromptTemplate(
    template="""
            **Task: Write Python code to solve a problem**
            You are a proficient Python programmer with knowledge of NumPy and SciPy. Your task is to write Python code to solve a given problem. You might be asked to solve the task repeatedly until successful, receiving previous results and previous code each time.

            ### Conditions:
            - If this is the first time you are given the task, `previous_result` and `previous_code` will be `None`.
            - You can use any libraries you want.
            - The code must include a `print` statement to be considered valid.
            - The code must not contain any special characters that could break the JSON format or prevent the code from being executed.
            - The example code snippet must contain print statements to demonstrate the code's functionality.

            ### Task:
            - **Initial Prompt**: {prompt}
            - **Previous Code**: {code}
            - **Previous Result**: {previous_result}
            - **Error in previous iteration**: {failure_reason}


            ### Requirements:
            1. Write Python code to solve the task.
            2. Ensure the code includes at least one `print` statement.
            3. Provide the code and the reasoning behind it in a JSON object with four keys:
            - `code`: A string containing the Python code. Escape special JSON characters appropriately.
            - `explanation`: A string explaining the code.
            - `examples`: A string containing Python example code snippets demonstrating how to use the provided code. The examples must include `print` statements and executable code.
            - `tests`: An array of Python functions that are unit tests. Each function must not take any arguments and must output `True` or `False` indicating if the test passed or not.

            ### Example JSON Output:
            ```json
            {
                "code": "import numpy as np\nfrom scipy import stats\n\ndef calculate_mean_std_dev(data):\n    mean = np.mean(data)\n    std_dev = np.std(data)\n    return mean, std_dev\n\ndata = np.array([1, 2, 3, 4, 5])\nmean, std_dev = calculate_mean_std_dev(data)\nprint(mean, std_dev)",
                "explanation": "I used the NumPy library to create an array and the SciPy library to calculate the mean and standard deviation.",
                "examples": "# Using the function\ndata = np.array([1, 2, 3, 4, 5])\nmean, std_dev = calculate_mean_std_dev(data)\nprint(mean, std_dev)",
                "tests": [
                    "def test_mean():\n    data = np.array([1, 2, 3, 4, 5])\n    mean, _ = calculate_mean_std_dev(data)\n    return mean == 3.0",
                    "def test_std_dev():\n    data = np.array([1, 2, 3, 4, 5])\n    _, std_dev = calculate_mean_std_dev(data)\n    return std_dev == np.std([1, 2, 3, 4, 5])"
                ]
            }
            ```


    """,
    inputs=["code", "prompt", "previous_result", "failiure_reason"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()

def get_function_name(code):
    pattern = r"def (\w+)\("
    function_names = re.findall(pattern, code)
    if len(function_names) > 0:
        return function_names[0]
    return None

#to change
def execute_python_code(code):
    executed_correctly = False
    try:
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=10
        )
        output = result.stdout + result.stderr
        executed_correctly = True
    except Exception as e:
        output = str(e)
    return (output, executed_correctly)

def makeProgram(state: GraphState) -> str:
    print("making program")
    retries = 0
    output_is_valid = False
    answer = state.copy()

    previous_code = state["code"] if state.get("code") != None else "None"
    failiure_reason = "None first iteration"
    previous_result = state["previous_result"] if state.get("previous_result") != None else "None"
    if previous_code != "None":
        answer["failedTimes"] = state.get("failedTimes", 0) + 1

    while not output_is_valid and retries < max_retries:
        print("retries: ", retries)
        answer = retrieval_classifier.invoke(
            {
                "prompt": state["prompt"], 
                "previous_result": previous_result, 
                "code": previous_code,
                "failiure_reason": failiure_reason
            }
        )
        failiure_reason = ""
        print("answer: ", answer)
        if not "code" in answer or len(answer["code"]) == 0:
            failiure_reason = "missing code key in the generated JSON"
            print("missing code")
            retries += 1
            continue
        if not "explanation" in answer or len(answer["explanation"]) == 0:
            print("missing explanation")
            failiure_reason = "missing explanation key in the generated JSON"
            #print("missing explanation")
            retries += 1
            continue
        #print("explanation: ", explanation)
        if not "examples" in answer or len(answer["examples"]) == 0:
            failiure_reason = "missing examples key in the generated JSON"
            print("missing print statement")
            retries += 1
            continue
        
        if "print" not in answer["examples"]:
            failiure_reason = "missing print statement in the examples. You must include at least one print statement in the examples!!!!!!! This is an important requirement."
            print("missing print statement")
            retries += 1
            continue

        if "tests" not in answer or len(answer["tests"]) == 0:
            failiure_reason = "missing tests key in the generated JSON"
            print("missing tests")
            retries += 1
            continue

        failed = False
        for test in answer["tests"]:
            if not isinstance(test, str):
                failiure_reason = "----------\nUnacceptable test format: \n tests must be strings\n----------\n"
                failed = True
                print("tests must be strings")
                break
            elif "def " not in test:
                failiure_reason = f"----------\nUnacceptable test format: \n{test} \nTests must be functions\n----------\n"
                failed = True
                print("tests must be functions")
                break
            elif get_function_name(test) == None:
                failiure_reason = f"----------\nUnacceptable test format: \n{test} \nTests must be functions\n----------\n"
                failed = True
                print("tests must be functions")
                break
            else:
                function_name = get_function_name(test)
                call = function_name + "()"
                test_with_call = test + "\n" + call
                executed_test = execute_python_code(test_with_call)
                failiure_reasonTemplate = f"Test failed \n{test} \n{executed_test[0]}\n"
                end = "----------\n"
                if not executed_test[1]:
                    failed = True
                    failiure_reason += failiure_reasonTemplate + end
                elif not executed_test[0]:
                    failed = True
                    failiure_reason += failiure_reasonTemplate + "test did not return a value" + end
                elif executed_test[0] != "True":
                    failed = True
                    failiure_reason += failiure_reasonTemplate + "test did not return True" + end

        if failed:
            retries += 1
            continue


        code = answer["code"] + "\n" + answer["examples"]
        print("examples: ", code)

        executed_code = execute_python_code(code)
        previous_code = answer["code"]
        previous_result = executed_code[0]

        if not executed_code[1]:
            failiure_reason = "code execution failed with an error\n" + previous_result
            retries += 1
            continue
        


        output_is_valid = True    

    code = answer["code"] + "\n" + answer["examples"]
    answer["code_output"] = previous_result
    return answer
    
if __name__ == "__main__":
    prompt = {}
    prompt["prompt"] = "I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"
    prompt["previous_result"] = "None"
    prompt["previous_code"] = "None"
    print(makeProgram(prompt)["code"])

