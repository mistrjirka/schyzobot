import hashlib
import os
import json
from langchain_community.utilities import SearxSearchWrapper
from .ollama_model import llm, llmNoJson  # Ensure this is correctly imported
from langchain_chroma import Chroma
from .graph_state import GraphState
from .webLoader import load_and_split_websites
from .embeddings import embeddings
from sentence_transformers import SentenceTransformer
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from .sourceClassifier import grade_document
import re

HASHES_FILE = 'document_hashes.json'

promptSearchQuery = PromptTemplate(
    template="""You are a search engine query generator. Given the following user question, create a relevant search engine query.

Question: {question}

The result should be a string containing the search query. No explanation or preamble is needed. Your answer will go straight into the search engine. So include just 1 best query. Do not include nay specification about what is the query.
Example input and output:
Input Question: "What is the capital of France?"
Output: "capital of France"
Input Question: "What is the population of India?"
Output: "population of India"
Input Question: "How to calculate fast fourier transform?"
Output: "fast fourier transform calculation"

""",
input_variables=["question"]
)
searchQueryCreator = promptSearchQuery | llmNoJson | StrOutputParser()
    
def extract_links_from_prompt(prompt_text):
    # Use regex to find all links in the prompt text
    links = re.findall(r'(https?://\S+)', prompt_text)
    return links

def hash_document(doc):
    doc_content = doc.page_content
    return hashlib.md5(doc_content.encode('utf-8')).hexdigest()

def load_hashes(file_path):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            return set(json.load(file))
    return set()

def save_hashes(file_path, hashes):
    with open(file_path, 'w') as file:
        json.dump(list(hashes), file)
        
def create_search_query(prompt: GraphState):
    query = searchQueryCreator.invoke({"question": prompt["prompt"]})  
    print("Search query: " + query)
    return query
    
    
    

def process_graph_state(graph_state: GraphState) -> GraphState:
    search = SearxSearchWrapper(searx_host="http://192.168.1.14:81", unsecure=True)  # Replace with your SearxNG instance URL
    chroma_db = Chroma(collection_name="resources", persist_directory="./chroma_data", embedding_function=embeddings)

    prompt = graph_state['prompt']    
    search_results = search.results(query=create_search_query(graph_state), num_results=18)
    extracted_links = extract_links_from_prompt(prompt)
    extracted_search_results = [{"link": link} for link in extracted_links]
            
    search_results += extracted_search_results
    relevant_resources = []
    
    split_docs = load_and_split_websites(search_results)

    existing_hashes = load_hashes(HASHES_FILE)
    new_docs = []
    new_hashes = set()

    for doc in split_docs:
        doc_hash = hash_document(doc)
        if doc_hash not in existing_hashes:
            new_docs.append(doc)
            new_hashes.add(doc_hash)

    if new_docs:
        chroma_db.add_documents(new_docs)
        existing_hashes.update(new_hashes)
        save_hashes(HASHES_FILE, existing_hashes)

    retriever = chroma_db.as_retriever(search_kwargs={"k": 28})
    results = retriever.invoke(graph_state["prompt"])
    indx = 0
    while len(relevant_resources) < 6 and indx < len(results):
        if grade_document(graph_state["prompt"], results[indx]):
            relevant_resources.append(results[indx])
        indx += 1

    graph_state['additionalResources'] = relevant_resources
    #print("Relevant resources:" + str(relevant_resources))
    
    return graph_state

if __name__ == "__main__":
    graph_state = GraphState(
        prompt="I want to solve fitting a linear regression model to a dataset. I have it here [(1.0, 2.5), (2, 3), (3, 4), (4, 5), (5, 6)]. Can you find parameters for me?",
        answer="",
        code="",
        code_output="",
        examples="",
        explanation="",
        failedTimes=0,
        type="example",
        additionalResources=[]
    )

    updated_graph_state = process_graph_state(graph_state)
    print(updated_graph_state)
