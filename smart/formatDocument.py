from langchain_core.documents import Document

def document_formatter(sources: list[Document]) -> str:
    documentsBySource = dict()

    if(len(sources) == 0):
        return "No documents found."

    for source in sources:
        if source.metadata["source"] not in documentsBySource:
            documentsBySource[source.metadata["source"]] = []
        documentsBySource[source.metadata["source"]].append(source)

    # Sort documents by start_index within each source
    for source in documentsBySource:
        documentsBySource[source].sort(key=lambda doc: doc.metadata.get('start_index', 0))

    # Concatenate the contents after sorting
    documentsFormatted = "\n".join(
        [f"Content of {source}:\n" + "\n".join([doc.page_content for doc in documentsBySource[source]]) 
         for source in documentsBySource]
    )

    return documentsFormatted
