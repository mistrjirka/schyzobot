from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser


from graph_state import GraphState
from ollama_model import llm
max_retries = 4

promptEN = PromptTemplate(
    template="""
        You are a smart programmer and you are given a task to solve and it is possible you will be given the task repeatitly until you succeed so you will be given previous result and previous code. 
        If the task is given to you for the first time the previous result will be None and previous code will be none.
        You can program in Python. You have the knowledge of numpy and scipy.

        Here is the task:
        {prompt}

        previous_code: 
        {previous_code}

        Here is the previous result:
        {previous_result}
        
        You have to write a Python code that solves the task. You can use any libraries you want. 
        
        You need to write example code that solves the task and print the result. If the result code does not contain print statement it will be considered as invalid!
        
        Provide the code and reasoning behind it as a JSON with a two key 'code' and 'explanation'. 
        
        The 'code' json key will contain string with the python code if you need to use for example " character you can use backslash to escape it. Any special json characters should be escaped.
        The explanation is a string that explains the code. The code part needs to be executable and non empty. 
    """,
    
    inputs=["prompt", "previous_result"]
)

retrieval_classifier = promptEN | llm | JsonOutputParser()


def makeProgram(prompt: dict) -> str:
    retries = 0
    output_is_valid = False
    answer = None
    while not output_is_valid and retries < max_retries:
        #print("retries: ", retries)
        answer = retrieval_classifier.invoke({"prompt": prompt["prompt"], "previous_result": prompt["previous_result"], "previous_code": prompt["previous_code"]})
        #print("answer: ", answer)
        if not "code" in answer:
            #print("missing code")
            retries += 1
            continue
        code = answer["code"]
        #print("code: ", code)
        if "explanation" not in answer:
            #print("missing explanation")
            retries += 1
            continue
        explanation = answer["explanation"]
        #print("explanation: ", explanation)
        if "print" not in code:
            #print("missing print statement")
            retries += 1
            continue

        output_is_valid = True    

    return answer
    
if __name__ == "__main__":
    prompt = {}
    prompt["prompt"] = "I want to solve fitting a linear regression model to a dataset. I have it here [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?"
    prompt["previous_result"] = "None"
    prompt["previous_code"] = "None"
    print(makeProgram(prompt)["code"])

