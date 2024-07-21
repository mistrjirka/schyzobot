import re
import requests
import time
import subprocess
import sys
import io
from colorama import init, Fore, Style

init(autoreset=True)
class Role:
    def __init__(self, system, assistant, url, grammar):
        self.system = system
        self.assistant = assistant
        self.url = url
        self.grammar = grammar
    
def prompt_splitter(list_of_prompts: list[(str,str)], start_prompt, message_prompt, roles: tuple[Role, Role], active=0): # list of tuples where first element is the role and second is the prompt
    #prompt format has {system}, {user}, {role}, {prompt}
    result = start_prompt.replace("{system}",roles[active].system)
    
    for role, prompt in list_of_prompts:
        found = False
        for role_index in range(len(roles)):
            if role == roles[role_index].assistant:
                found = True
                break
        if not found:
            raise ValueError("Role not found in roles")

        if role != roles[active].system:
            result += message_prompt.replace("{role}", "user").replace("{prompt}", prompt)
        else:
            result += message_prompt.replace("{role}", "assistant").replace("{prompt}", prompt)
    result += message_prompt.replace("{role}", "assistant").replace("{prompt}", "")

    return result


def generate_response(prompt, llama_endpoint, n_predict=1024, temperature=0.5, stop_tokens=None, grammar=None):
    api_data = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stop": stop_tokens,
        "tokens_cached": 0,
        "repeat_penalty": 1.2,
        "grammar": grammar
    }

    retries = 5
    backoff_factor = 1
    while retries > 0:
        try:
            response = requests.post(llama_endpoint, headers={"Content-Type": "application/json"}, json=api_data)
            json_output = response.json()
            output = json_output['content']
            break
        except:
            print(Fore.RED + "My AI model is not responding, try again in a moment 🔥 🐳")
            time.sleep(backoff_factor)
            
            backoff_factor *= 2
            retries -= 1
            output = "My AI model is not responding, try again in a moment 🔥🐳"
            continue

    return output


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

def is_executable_code(code):
    # A simple heuristic to determine if the code contains executable statements
    return any(keyword in code for keyword in ['print(', 'return', 'input(', 'assert', 'if ', 'for ', 'while ', 'def ', 'class '])


personalities = {
    "programmer": """You are a helpful programmer capable of writing and executing Python code. you have the capability to program in python. If you create a code block. You need to put print with the shown functionality at the 
    end to show to the manager that the code works. Write examples that test the created functions. So write something like 
    ```python \n def your_created_function(): \n return 10 \n print(your_created_function()) \n ```""",
    "manager": "You are a manager who can guide and provide high-level overviews but does not write code. Your task is to steer the programmer. You will se what the programmer had written and you can see his errors. Tell him what is wrong and what he should do. Under no circumstances should you write code. Programmer often forgets to make example and print statement at the end. Look at the code and if there is no print statement at the end, tell him to add it. If there is no example, tell him to add it. If there is an error, tell him what is wrong. If the code is correct, tell him that the code is correct.",
    "stroke_checker": "You are a stroke checker. You are responsible for checking that the response is not a bug a some continous string of nonsense. You are basically looking if the response is valid makes sense and if not you say no, if the response makes sense you say yes. You are the last line of defense."
}

start_prompt = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>{system}<|eot_id|>"""
middle_prompt = """<|eot_id|><|start_header_id|>{role}<|end_header_id|>{prompt}"""

# Define initial roles for the interaction


llama_endpoint = "http://localhost:8086/completion"  # Replace with your actual endpoint
programmer_endpoint = "http://localhost:8086/completion"  # Replace with your actual endpoint

roles = [
    Role(personalities["programmer"], "programmer", programmer_endpoint, """
        root ::= text codeBlock text

            # Code block rule
            codeBlock ::= "```" lang "\n" code "\n```" "\n"

            # Language identifier
            lang ::= [a-zA-Z0-9]+

            # Code content (excluding triple backticks)
            code ::= ( [^`] )*

            # Text content (excluding triple backticks)
            text ::= ( [^`] )*

            # Whitespace
            ws ::= [ \t\n]*

    """),
    Role(personalities["manager"], "manager", llama_endpoint,
        """
            # Root rule 0 or 1 code block
            root ::= text (codeBlock)?
            lang ::= [a-zA-Z0-9]+

            codeBlock ::= "```" lang "\n" code "\n```" "\n"

            # Text content (excluding triple backticks)
            text ::= ( [^`] )*
            code ::= ( [^`] )*

            # Whitespace
            ws ::= [ \t\n]*
        """
    )
]

stroke_checker = Role(personalities["stroke_checker"], "stroke_checker", llama_endpoint, """
                      # root rule is yes string or no string
                        root ::= yes | no
                        yes ::= "yes"
                        no ::= "no"
                    """)

user_prompt = ("manager","""Create a function that takes list of functions as input and data and fits the function using LSTSQ to the data. You can use numpy (by importing numpy)""")

conversation_log = [user_prompt]
boredom_threshold = 3
boredom_counter = 0
current_role_index = 0

while True:
    to_send = prompt_splitter(conversation_log, start_prompt, middle_prompt, roles, current_role_index)
    print("Sending prompt to the AI model...", to_send)
    response = ""
    responseIsAStroke = True
    while responseIsAStroke:
        response = generate_response(to_send, 
            roles[current_role_index].url, stop_tokens=["<|eot_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>"],
            grammar=roles[current_role_index].grammar
        )
        code_response = "\nAttempting to execute code:\n"
        runned_code = False
        print("finished response")
        accumulated_code = ""

        # Check if there is any code block in the response
        code_block_matches = re.findall(r'```(.*?)```', response, re.DOTALL)

        for code_block in code_block_matches:
            print(Fore.YELLOW + "Computer: I found some code to execute. Let's see what it does!")
            
            # Extract the language if specified
            first_line = code_block.split('\n', 1)[0].strip().lower()
            if first_line in ['python', 'py']:
                language = first_line
                code = code_block[len(first_line):].strip()
            else:
                language = 'python'
                code = code_block.strip()
            
            # Check if the language is a variation of Python
            if language in ['python', 'py']:
                accumulated_code += "\n" + code
            else:
                code_response = f"{Fore.RED}Cannot run code written in {language}.\n"

        if(accumulated_code != ""):
            runned_code = True
            print(Fore.YELLOW + "Bot: Executing the code...")
            print(Fore.YELLOW + f"Code to execute: {accumulated_code}")
            execution_result = execute_python_code(accumulated_code)
            code_response = f"\n{Fore.GREEN}Code execution result:\n{execution_result}\n"
        if runned_code:
            print(Fore.GREEN + "Bot: Code executed successfully.")
            response = response + code_response
        
        print(Fore.CYAN + "Bot: " + response)
        print(Fore.MAGENTA + "-"*50)  # Visual separator

        # Check if the response is a stroke
        stroke_check_prompt = prompt_splitter([(roles[current_role_index].assistant, response)], start_prompt, middle_prompt, [stroke_checker,roles[current_role_index]])
        print("Sending prompt to the AI model...", stroke_check_prompt)
        stroke_check_response = generate_response("Does this answer make sense? Arent there some long strings of nonesense? If there is a nonsense answer no if there is not nonsense answer yes.\nChatbot response:\n"+stroke_check_prompt, stroke_checker.url, stop_tokens=["<|eot_id|>", "<|end_header_id|>", "<|eot_id|>", "<|end_of_text|>"], grammar=stroke_checker.grammar)
        print(Fore.CYAN + "Bot: " + stroke_check_response)
        responseIsAStroke = stroke_check_response == "no"


    conversation_log.append((roles[current_role_index].assistant, response))
    
    # Rotate roles for the next interaction
    current_role_index = (current_role_index + 1) % len(roles)
    
    # Simple boredom detection
    if len(conversation_log) > 3 and response in conversation_log[-4:-1]:
        boredom_counter += 1
    else:
        boredom_counter = 0
    
    if boredom_counter >= boredom_threshold:
        new_topic = generate_response("Give me a new topic to think about.", llama_endpoint, stop_tokens=["\nuser:", "\nsystem:", "\nassistant:"])
        print(Fore.YELLOW + f"Bot is bored. New topic: {new_topic}")
        conversation_log.append(new_topic)
        boredom_counter = 0
