from langchain_community.chat_models import ChatOllama

# select ollama model
MODEL = "dolphin-llama3:8b-v2.9-fp16"#"dolphin-llama3:70b"#"dolphin-llama3:8b-v2.9-q8_0"
# higher temparature means higher creativity
TEMPERATURE = 0.5

print("Loading ollama model...", end=" ", flush=True)
llm = ChatOllama(model=MODEL, temperature=TEMPERATURE, format="json")
print("DONE")