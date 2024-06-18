import re
import requests
import time
import subprocess
import sys
import io
from colorama import init, Fore, Style

init(autoreset=True)

def generate_response(prompt, llama_endpoint, n_predict=2048, temperature=0.7, stop=None):
    api_data = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temperature,
        "stop": stop,
        "tokens_cached": 0,
        "repeat_penalty": 1.2
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

initial_prompt = """Hi, you have capability to program in python 
                    if you create a code block you can execute code 
                    in it. You can then print the output of the code block. 
                    to get the result. I will give you a task and you will try to complete it. 
                    I want you to create raft in python. 
                    Create a distributed system simulator and try it."""
conversation_log = [initial_prompt]

boredom_threshold = 3
boredom_counter = 0
llama_endpoint = "http://localhost:8086/completion"  # Replace with your actual endpoint

accumulated_code = ""

while True:
    prompt = " ".join(conversation_log[-3:])  # Use the last few exchanges as context
    response = generate_response(prompt, llama_endpoint)
    code_response = "\nAttempting to execute code:\n"
    runned_code = False
    
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
            if is_executable_code(code):
                runned_code = True
                execution_result = execute_python_code(accumulated_code)
                code_response = f"\n{Fore.GREEN}Code execution result:\n{execution_result}\n"
                accumulated_code = ""  # Reset after execution
        else:
            code_response = f"{Fore.RED}Cannot run code written in {language}.\n"
    
    if runned_code:
        response = response + code_response
    
    print(Fore.CYAN + "Bot: " + response)
    print(Fore.MAGENTA + "-"*50)  # Visual separator

    conversation_log.append(response)
    
    # Simple boredom detection
    if len(conversation_log) > 3 and response in conversation_log[-4:-1]:
        boredom_counter += 1
    else:
        boredom_counter = 0
    
    if boredom_counter >= boredom_threshold:
        new_topic = generate_response("Give me a new topic to think about.", llama_endpoint)
        print(Fore.YELLOW + f"Bot is bored. New topic: {new_topic}")
        conversation_log.append(new_topic)
        boredom_counter = 0
