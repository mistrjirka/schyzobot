from langchain_community.utilities import SearxSearchWrapper
from ollama_model import llm  # Ensure this is correctly imported
from langchain_chroma import Chroma
from graph_state import GraphState
from sentence_transformers import SentenceTransformer
from typing import List
import bs4

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_community.document_loaders import WebBaseLoader
import re

# load embedding model as sentence tranformer (model will be automatically downloaded from hugging face)
class MyEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    # this function is needed for Chroma embeddings
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()


### static
promptEN = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n 
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score true or false in form of JSON with key "answer" score to indicate whether the document is relevant to the question without preamble or explanation.
    Example: {{"answer": "YES"}} or {{"answer": "NO"}} \n
    """,
    input_variables=["question", "document"],
) 
retrieval_grader = promptEN | llm | JsonOutputParser()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
embeddings = MyEmbeddings()


### static


def grade_document(question, document):
    answer = retrieval_grader.invoke({"question": question, "document": document})
    while "answer" not in answer or answer["answer"] not in ["YES", "NO"]:
        answer = retrieval_grader.invoke({"question": question, "document": document})

    print(f"Grading document: {answer["answer"]}")
    if answer["answer"] == "YES":
        return True
    else:
        return False
    
    
def load_and_split_websites(search_results: List[dict]):
    urls = [result['link'] for result in search_results]
    web_loader = WebBaseLoader(web_paths=urls, bs_kwargs=dict(parse_only=bs4.SoupStrainer("main")))
    
    docs = web_loader.load()
    print(f"{len(docs)} documents loaded")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    return text_splitter.split_documents(docs)
    
    

def process_graph_state(graph_state: GraphState) -> GraphState:
    # Initialize the SearxSearchWrapper
    search = SearxSearchWrapper(searx_host="http://192.168.1.14:81", unsecure=True)  # Replace with your SearxNG instance URL

    # Initialize the Chroma vector database
    chroma_db = Chroma(collection_name="resources", persist_directory="./chroma_data", embedding_function=embeddings)

    # Extract the prompt from the graph state
    prompt = graph_state['prompt']
    
    

    # Perform the search using SearxSearchWrapper
    search_results = search.results(query=prompt, num_results=10)

    # Process the search results
    relevant_resources = []
    
    split_docs = load_and_split_websites(search_results)
    
    chroma_db.add_documents(split_docs)
        
    
    # get documents from chroma by querying the prompt
    retriever = chroma_db.as_retriever(search_kwargs={"k": 10})
    results = retriever.invoke(graph_state["prompt"])
    indx = 0
    while len(relevant_resources) < 5 and indx < len(results):
        if grade_document(graph_state["prompt"], results[indx]):
            relevant_resources.append(results[indx])
        
        indx += 1
            
    # Update the GraphState with the relevant resources
    graph_state['additionalResources'] = relevant_resources
    print("Relevant resources:" + str(relevant_resources))
    
    return graph_state
if __name__ == "__main__":
    # Example usage
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
