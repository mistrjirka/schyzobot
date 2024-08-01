from .graph_state import GraphState
from .webLoader import load_and_split_websites
from .embeddings import embeddings
from langchain_chroma import Chroma
from .sourceClassifier import grade_document
def load_links(graph_state: GraphState) -> GraphState:
    print("extractLinksDataNODE")
    urls = [{"link": link} for link in graph_state["links"]]
    
    docs = load_and_split_websites(urls)
    
    chroma_db = Chroma(collection_name="resources", embedding_function=embeddings)
    chroma_db.add_documents(docs)
    retriever = chroma_db.as_retriever(search_kwargs={"k": 50})
    results = retriever.invoke(graph_state["prompt"])
    indx = 0

    relevant_resources = []

    while len(relevant_resources) < 6 and indx < len(results):
        if grade_document(graph_state["prompt"], results[indx]):
            relevant_resources.append(results[indx])
        indx += 1


    graph_state["additionalResources"] = relevant_resources
    
    return graph_state
