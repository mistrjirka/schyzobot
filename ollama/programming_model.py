from langchain_community.chat_models import ChatOllama

# select ollama model
MODEL = "granite-code"
#"deepseek-coder-v2:16b" "codegemma:7b-code-q5_0" "granite-code"
# higher temparature means higher creativity
TEMPERATURE = 0.5

print("Loading ollama model...", end=" ", flush=True)
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, format="json")
print("DONE")

