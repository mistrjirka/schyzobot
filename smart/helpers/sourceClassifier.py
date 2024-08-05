from smart.models.ollama_model import llm, llmNoJson, llmBinary  # Ensure this is correctly imported
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from smart.helpers.graph_state import GraphState
from langchain_core.prompts import ChatPromptTemplate


promptEN_summarization = PromptTemplate(
    template="""   
Chat with the user:
{all_messages}

You are a grader assessing the relevance of a retrieved document for summarization.
 
Here is the retrieved document: \n\n {document} \n\n
Assess the document for relevance to the core content, filtering out irrelevant parts like headers, footers, and unrelated links. If the document contains substantial content suitable for summarization, grade it as relevant.
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Answer with a binary score "YES" or "NO", any additional information will be considered as a wrong answer. \n
    """,
    input_variables=["document", "all_messages"],
) 

    
promptChromaQuery = PromptTemplate(
    template="""
Previous conversation with the user:
{all_messages}

Question: {question}

You taskl is to extract important keywords and context from the question. All of the keywords need to be extracted. Longer is better..
The result should be a string containing the search query. No explanation or preamble is needed. Your answer will go straight into the search engine. So include just 1 best query. Do not include nay specification about what is the query. The query should be realitvely long and contain the relevant keywords for each part of the question and the relevant context.
Example input and output:
Input Question: "What is the capital of France?"
Output: "capital of France"
Input Question: "What is the population of India?"
Output: "population of India"
Input Question: "How to calculate fast fourier transform?"
Output: "fast fourier transform calculation"

""",
input_variables=["question", "all_messages"]
)

promptEN = PromptTemplate(
    template="""
Chat with the user:
{all_messages}

You are a grader assessing relevance of a retrieved document to a user question. \n 
Here is the retrieved document: \n\n {document} \n\n
Here is the user question: {question} \n
If the document contains keywords related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Answer with a binary score "YES" or "NO", any additional information will be considered as a wrong answer. \n
    """,
    input_variables=["question", "document", "all_messages"],
) 

retrieval_grader = promptEN | llmBinary | StrOutputParser()

summarization_grader = promptEN_summarization | llmBinary | StrOutputParser()

chroma_query_creator = promptChromaQuery | llmNoJson | StrOutputParser()
chroma_query_creator_creative = promptChromaQuery | llmNoJson | StrOutputParser()

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

def grade_document(question, document, all_messages):
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    answer = retrieval_grader.invoke({"question": question, "document": document, "all_messages": promptChat})
    answer = answer.upper()
    retries = 0
    while "YES" not in answer and "NO" not in answer and retries < 10:
        print("bad answer: ", answer)
        print(f"Retrying grading document: {retries}")
        retries += 1
        answer = retrieval_grader.invoke({"question": question, "document": document, "all_messages": promptChat})
    print(f"Grading document: {answer}")

    return "YES" in answer

def grade_for_summarization(document, all_messages):
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    answer = summarization_grader.invoke({"document": document, "all_messages": promptChat})
    answer = answer.upper()
    retries = 0
    while "YES" not in answer and "NO" not in answer and retries < 10:
        print("bad answer: ", answer)
        print(f"Retrying grading document: {retries}")
        retries += 1
        answer = summarization_grader.invoke({"document": document, "all_messages": promptChat})

    print(f"Grading document: {answer}")
    return "YES" in answer

def create_chroma_query(prompt: GraphState, creative=False):
    all_messages = prompt["messages"]
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    if creative:
        query = chroma_query_creator_creative.invoke({"question": prompt["prompt"], "all_messages": promptChat})
    else:
        query = chroma_query_creator.invoke({"question": prompt["prompt"], "all_messages": promptChat})  
    
    return query
