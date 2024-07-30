from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pyston import PystonClient, File
import subprocess
import sys
import asyncio
from .graph_state import GraphState
from .ollama_model import llm
import re
import os
max_retries = 5

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
{{
    "code": "
import numpy as np
from scipy import stats

def calculate_mean_std_dev(data):
    mean = np.mean(data)
    std_dev = np.std(data)
    return mean, std_dev


data = np.array([1, 2, 3, 4, 5])
mean, std_dev = calculate_mean_std_dev(data)
print(mean, std_dev)",
    "explanation": "I used the NumPy library to create an array and the SciPy library to calculate the mean and standard deviation.",
    "examples": "# Using the function\ndata = np.array([1, 2, 3, 4, 5])\nmean, std_dev = calculate_mean_std_dev(data)\nprint(mean, std_dev)",
    "tests": [
"
def test_mean():
    data = np.array([1, 2, 3, 4, 5])
    mean, _ = calculate_mean_std_dev(data)
    return mean == 3.0",
"
def test_std_dev():
    data = np.array([1, 2, 3, 4, 5])
    something, std_dev = calculate_mean_std_dev(data)
    return std_dev == np.std([1, 2, 3, 4, 5])"
    ]
}}
```


    """,
    inputs=["code", "prompt", "previous_result", "failure_reason"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()

def get_function_name(code):
    pattern = r"def (\w+)\("
    function_names = re.findall(pattern, code)
    if len(function_names) > 0:
        return function_names[0]
    return None

async def async_execute_python_code(code):
    executed_correctly = False
    client = PystonClient()
    result = await client.execute("python", [File(code)])
    print("output from execution: ", result.run_stage.output)
    output = result.run_stage.output
    executed_correctly = result.run_stage.code == 0
    return (output, executed_correctly)

def execute_python_code(code):
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(async_execute_python_code(code))


def makeProgram(state: GraphState) -> str:
    print("making program")
    retries = 0
    output_is_valid = False
    answer = state.copy()

    previous_code = state["code"] if state.get("code") != None else "None"
    failure_reason = "None first iteration"
    previous_result = state["previous_result"] if state.get("previous_result") != None else "None"
    if previous_code != "None":
        answer["failedTimes"] = state.get("failedTimes", 0) + 1

    while not output_is_valid and retries < max_retries:
        #if(retries > 0):
        #    state["update_process"]("Code failed retrying: " + str(answer["failedTimes"]) + "\n")
        print("retries: ", retries)
        answer = retrieval_classifier.invoke(
            {
                "prompt": state["prompt"], 
                "previous_result": previous_result, 
                "code": previous_code,
                "failure_reason": failure_reason
            }
        )
        failure_reason = ""
        print("answer: ", answer)
        if not "code" in answer or len(answer["code"]) == 0:
            failure_reason = "missing code key in the generated JSON"
            print("missing code")
            retries += 1
            continue
        if not "explanation" in answer or len(answer["explanation"]) == 0:
            print("missing explanation")
            failure_reason = "missing explanation key in the generated JSON"
            #print("missing explanation")
            retries += 1
            continue
        #print("explanation: ", explanation)
        if not "examples" in answer or len(answer["examples"]) == 0:
            failure_reason = "missing examples key in the generated JSON"
            print("missing examples")
            retries += 1
            continue
        
        if "print" not in answer["examples"]:
            failure_reason = "missing print statement in the examples. You must include at least one print statement in the examples!!!!!!! This is an important requirement."
            print("missing print statement")
            retries += 1
            continue

        if "tests" not in answer or len(answer["tests"]) == 0:
            failure_reason = "missing tests key in the generated JSON"
            print("missing tests")
            retries += 1
            continue

        failed = False
        for test in answer["tests"]:
            if not isinstance(test, str):
                failure_reason = "----------\nUnacceptable test format: \n tests must be strings\n----------\n"
                failed = True
                print("tests must be strings")
                break
            elif "def " not in test:
                failure_reason = f"----------\nUnacceptable test format: \n{test} \nTests must be functions\n----------\n"
                failed = True
                print("tests must be functions")
                break
            elif get_function_name(test) == None:
                failure_reason = f"----------\nUnacceptable test format: \n{test} \nTests must be functions\n----------\n"
                failed = True
                print("tests must be functions")
                break
            else:
                function_name = get_function_name(test)
                call = function_name + "()"
                test_with_call = f"""{answer["code"]}\n{test}\nprint("___test result: " + str({call}) + "___")"""
                print("test_with_call: \n", test_with_call)
                
                executed_test = execute_python_code(test_with_call)
                failure_reasonTemplate = f"Test failed \n{test} \n{executed_test[0]}\n"
                end = "----------\n"
                if not executed_test[1]:
                    failed = True
                    failure_reason += failure_reasonTemplate + end
                elif not executed_test[0]:
                    failed = True
                    failure_reason += failure_reasonTemplate + "test did not return a value" + end
                elif not "___test result: True___" in executed_test[0]:
                    failed = True
                    failure_reason += failure_reasonTemplate + "test did not return True" + end

        if failed:
            print("failed tests")
            print(failure_reason)
            retries += 1
            continue


        code = answer["code"] + "\n" + answer["examples"]
        print("examples: ", code)

        executed_code = execute_python_code(code)
        previous_code = answer["code"]
        previous_result = executed_code[0]

        if not executed_code[1]:
            failure_reason = "code execution failed with an error\n" + previous_result
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

