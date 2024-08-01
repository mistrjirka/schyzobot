from .ollama_model import llm, llmNoJson  # Ensure this is correctly imported
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import JsonOutputParser
from langchain.prompts import PromptTemplate

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

def grade_document(question, document):
    answer = retrieval_grader.invoke({"question": question, "document": document})
    while "answer" not in answer or answer["answer"] not in ["YES", "NO"]:
        answer = retrieval_grader.invoke({"question": question, "document": document})

    print(f"Grading document: {answer['answer']}")
    return answer["answer"] == "YES"