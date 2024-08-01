from sentence_transformers import SentenceTransformer
from typing import List
class MyEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()


embeddings = MyEmbeddings()
