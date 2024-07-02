from langchain_community.chat_models import ChatOllama

# select ollama model
MODEL = "qwen2:0.5b-instruct-q2_K"
# higher temparature means higher creativity
TEMPERATURE = 0.5

print("Loading ollama model...", end=" ", flush=True)
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE)
print("DONE")

