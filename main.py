import requests
import time

def generate_response(prompt, llama_endpoint, n_predict=150, temperature=0.7, stop=None):
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

initial_prompt = "What do you think about the future of AI?"
conversation_log = [initial_prompt]

boredom_threshold = 3
boredom_counter = 0
llama_endpoint = "http://localhost:8086/completion"  # Replace with your actual endpoint

while True:
    prompt = " ".join(conversation_log[-3:])  # Use the last few exchanges as context
    response = generate_response(prompt, llama_endpoint)
    print(f"Bot: {response}")
    
    conversation_log.append(response)
    
    # Simple boredom detection
    if len(conversation_log) > 3 and response in conversation_log[-4:-1]:
        boredom_counter += 1
    else:
        boredom_counter = 0
    
    if boredom_counter >= boredom_threshold:
        new_topic = generate_response("Give me a new topic to think about.", llama_endpoint)
        print(f"Bot is bored. New topic: {new_topic}")
        conversation_log.append(new_topic)
        boredom_counter = 0
