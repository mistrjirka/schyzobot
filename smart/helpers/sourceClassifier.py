from smart.models.ollama_model import llm, llmNoJson  # Ensure this is correctly imported
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain.prompts import PromptTemplate
from smart.helpers.graph_state import GraphState
from langchain_core.prompts import ChatPromptTemplate

promptEN = PromptTemplate(
    template="""
Chat with the user:
{all_messages}

You are a grader assessing relevance of a retrieved document to a user question. \n 
Here is the retrieved document: \n\n {document} \n\n
Here is the user question: {question} \n
If the document contains keywords related to the user question, grade it as relevant. \n
It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
Give a binary score true or false in form of JSON with key "answer" score to indicate whether the document is relevant to the question without preamble or explanation.
Example: {{"answer": true}} or {{"answer": false}} \n
    """,
    input_variables=["question", "document", "all_messages"],
) 

promptEN_summarization = PromptTemplate(
    template="""
Chat with the user:
{all_messages}

You are a grader assessing the relevance of a retrieved document for summarization.

Here is the retrieved document:

{document}

Assess the document for relevance to the core content, filtering out irrelevant parts like headers, footers, and unrelated links. If the document contains substantial content suitable for summarization, grade it as relevant.

Give a binary score true or false in the form of JSON with the key "answer" to indicate whether the document is relevant for summarization without preamble or explanation.

Example: {{"answer": true}} or {{"answer": false}}
    """,
    input_variables=["document", "all_messages"],
) 

    
promptChromaQuery = PromptTemplate(
    template="""
Previous conversation with the user:
{all_messages}

Question: {question}


You are a search Vector database query generator. Given the following user question, create a relevant Chroma vector database query.
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

retrieval_grader = promptEN | llm | JsonOutputParser()

summarization_grader = promptEN_summarization | llm | JsonOutputParser()

chroma_query_creator = promptChromaQuery | llmNoJson | StrOutputParser()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)

def grade_document(question, document, all_messages):
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    answer = retrieval_grader.invoke({"question": question, "document": document, "all_messages": promptChat})
    retries = 0
    while "answer" not in answer or answer["answer"] not in [True, False] and retries < 10:
        retries += 1
        answer = retrieval_grader.invoke({"question": question, "document": document, "all_messages": promptChat})

    print(f"Grading document: {answer['answer']}")
    return answer["answer"]

def grade_for_summarization(document, all_messages):
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    answer = summarization_grader.invoke({"document": document, "all_messages": promptChat})
    retries = 0
    while "answer" not in answer or answer["answer"] not in [True, False] and retries < 10:
        retries += 1
        answer = summarization_grader.invoke({"document": document, "all_messages": promptChat})

    print(f"Grading document: {answer['answer']}")
    return answer["answer"]

def create_chroma_query(prompt: GraphState):
    all_messages = prompt["messages"]
    promptChat = ChatPromptTemplate.from_messages(all_messages)
    query = chroma_query_creator.invoke({"question": prompt["prompt"], "all_messages": promptChat})  
    return query