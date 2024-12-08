from langchain_community.chat_models import ChatOllama

# select ollama model
MODEL = "llama3.1:70b-instruct-q2_K" #"dolphin-llama3:70b" 
MODEL_ALT1 = ""
#"dolphin-llama3:8b-v2.9-q8_0"
#"deepseek-coder-v2:16b" "codegemma:7b-code-q5_0" "granite-code"
# higher temparature means higher creativity
TEMPERATURE = 0.5

print("Loading ollama model...", end=" ", flush=True)
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, format="json")
llmNoJson = ChatOllama(model=MODEL, temperature=TEMPERATURE)
print("DONE")

