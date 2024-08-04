from .graph_state import GraphState
from .webLoader import load_and_split_websites
from .embeddings import embeddings
from langchain_chroma import Chroma
from .sourceClassifier import grade_for_summarization, create_chroma_query
def load_links(graph_state: GraphState) -> GraphState:
    print("extractLinksDataNODE")
    urls = [{"link": link} for link in graph_state["links"]]
    
    docs = load_and_split_websites(urls)
    
    chroma_db = Chroma(collection_name="resources", embedding_function=embeddings)
    if len(docs) > 0:
        chroma_db.add_documents(docs)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 50})
    chromaQuery = create_chroma_query(graph_state)
    results = retriever.invoke(chromaQuery)
    indx = 0

    relevant_resources = []
    all_messages = graph_state["messages"]


    while len(relevant_resources) < 7 and indx < len(results):
        if grade_for_summarization(results[indx], all_messages):
            relevant_resources.append(results[indx])
        indx += 1


    graph_state["additionalResources"] = relevant_resources
    
    return graph_state
