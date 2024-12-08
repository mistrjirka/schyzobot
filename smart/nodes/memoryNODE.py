import hashlib
import os
import json
from langchain_community.utilities import SearxSearchWrapper
from smart.models.ollama_model import llmNoJson, llmNoJsonCreative
from langchain_chroma import Chroma
from smart.helpers.graph_state import GraphState
from smart.helpers.webLoader import load_and_split_websites
from smart.models.embeddings import embeddings
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from smart.helpers.sourceClassifier import grade_document,create_chroma_query
from langchain_core.prompts import ChatPromptTemplate
from iteration_utilities import unique_everseen
import random
import math
import re

HASHES_FILE = 'document_hashes.json'

promptSearchQuery = PromptTemplate(
    template="""
Previous conversation with the user:
{all_messages}

Question: 
{question}

Previous query (try to create different original one)
{prev_query}
You are a search engine query generator. Given the following user question, create a relevant search engine query.

The result should be a string containing the search query. No explanation or preamble is needed. Your answer will go straight into the search engine. So include just 1 best query. Do not include nay specification about what is the query.
Example input and output:
Input Question: "What is the capital of France?"
Output: "capital of France"
Input Question: "What is the population of India?"
Output: "population of India"
Input Question: "How to calculate fast fourier transform?"
Output: "fast fourier transform calculation"

Remember DO NOT make just simple copy of the previous query. Try to create different original one.
The query needs to have the relevant keywords and context for the question. So if there is question about last expedition. The word last has to be there.
""",

input_variables=["question", "all_messages", "prev_query"]
)
promptSearchQueryCreative = PromptTemplate(
    template="""
Previous conversation with the user:
{all_messages}

Question: 
{question}

Previous query (try to create different original one)
{prev_query}

You are a creative search engine query generator. Given the following user question, create a relevant creative search engine query.

The result should be a string containing the search query. No explanation or preamble is needed. Your answer will go straight into the search engine. So include just 1 best query. Do not include nay specification about what is the query.
Example input and output:
Input Question: "What is the capital of France?"
Output: "capital of France"
Input Question: "What is the population of India?"
Output: "population of India"
Input Question: "How to calculate fast fourier transform?"
Output: "fast fourier transform calculation"
""",

input_variables=["question", "all_messages", "prev_query"]
)

searchQueryCreator = promptSearchQuery | llmNoJson | StrOutputParser()
searchQueryCreatorCreative = promptSearchQueryCreative | llmNoJsonCreative | StrOutputParser()

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
        
def create_search_query(prompt: GraphState, creative: bool = False, prev_query = "None, first time run") -> str:
    all_messages = prompt["messages"]
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    if creative:
        query = searchQueryCreatorCreative.invoke({"question": prompt["prompt"], "all_messages": promptChat, "prev_query": prev_query})
    else:
        query = searchQueryCreator.invoke({"question": prompt["prompt"], "all_messages": promptChat, "prev_query": prev_query})  
    #remove "" from the query
    query = query.replace('"', '')
    print("Search query: " + query)

    return query

def search_through_resources(graph_state: GraphState, chroma_db: Chroma, N: int, to_search: int):
    search = SearxSearchWrapper(searx_host="http://192.168.1.14:81", unsecure=True)  # Replace with your SearxNG instance URL
    prompt = graph_state['prompt']

    extracted_links = extract_links_from_prompt(prompt)
    extracted_search_results = [{"link": link} for link in extracted_links]

    queried = []

    search_results = extracted_search_results
    for i in range(N):
        if i == 0:
            queried.append(create_search_query(graph_state, False))
        else:
            queried.append(create_search_query(graph_state, i > 1, "\n".join(queried)))
        if "update_process" in graph_state:
            graph_state["update_process"](f"Searching for relevant resources {i+1}/{N} with query: {queried[-1]}\n")

    queried = list(set(queried))
    to_search_count = math.ceil(to_search / len(queried))

    for query in queried:
        found = search.results(query=query, num_results=to_search_count)
        search_results += found
            
    
    search_results = list(unique_everseen(search_results))
    print("found " + str(len(search_results)))
    
    split_docs = load_and_split_websites(search_results, max_docs=300)

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


__get_chroma_var = None
def get_chroma():
    global __get_chroma_var
    if __get_chroma_var is None:
        __get_chroma_var = Chroma(collection_name="resources", persist_directory="./chroma_data", embedding_function=embeddings)
    return __get_chroma_var


def process_graph_state(graph_state: GraphState) -> GraphState:
    if "update_process" in graph_state:
        graph_state["update_process"]("Entering memory processing phase...\n")
    else:
        print("update process not found")
    chroma_db = get_chroma()
    
    N = 2
    RELEVANT_SOURCES = 20
    to_search = 15

    if "update_process" in graph_state:
        graph_state["update_process"]("🌐 Searching through web resources...\n")
    
    search_through_resources(graph_state, chroma_db, N, to_search)
    
    if "update_process" in graph_state:
        graph_state["update_process"]("🔍 Generating search queries for local database...\n")
    
    queries = []
    for i in range(N):
        query = create_search_query(graph_state, i > 0, "\n".join(queries))
        queries.append(query)
        if "update_process" in graph_state:
            graph_state["update_process"](f"Generated query {i+1}: {query}\n")

    queries = list(set(queries))
    to_extract = math.ceil(100 / len(queries))

    if "update_process" in graph_state:
        graph_state["update_process"]("📚 Searching through local knowledge base...\n")

    retriever = chroma_db.as_retriever(search_kwargs={"k": to_extract})
    found = []
    for searchChroma in queries:
        found.append(retriever.invoke(searchChroma))

    zipped_queried = list(zip(*found))
    results = []
    for queried in zipped_queried:
        results.extend(queried)
    
    results = list(unique_everseen(results))
    indx = 0
    all_messages = graph_state["messages"]
    relevant_resources = []  # Initialize the list here
    if "update_process" in graph_state:
        graph_state["update_process"]("🔍 Grading search results...\n")
    
    total_results = min(RELEVANT_SOURCES, len(results))
    while len(relevant_resources) < RELEVANT_SOURCES and indx < len(results):
        if "update_process" in graph_state:
            # Use return character to update progress inline
            graph_state["update_process"](f"\rGrading document {indx + 1}/{total_results} ({len(relevant_resources)} relevant found)")
        
        if grade_document(graph_state["prompt"], results[indx], all_messages):
            relevant_resources.append(results[indx])
        indx += 1

    # Add a newline after the progress updates
    if "update_process" in graph_state:
        graph_state["update_process"]("\n")
    
    graph_state['additionalResources'] = relevant_resources
    
    if "update_process" in graph_state:
        graph_state["update_process"](f"Found {len(relevant_resources)} relevant resources\n")
    
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
