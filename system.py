import numpy as np
import json
import re
from typing import List, Dict, Tuple
from collections import Counter
from dataclasses import dataclass
import math

@dataclass
class Document:
    id: str
    title: str
    context: str
    embedding: np.ndarray = None

class SimpleEmbedder:

    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.is_fitted = False

    def preprocess(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^\w\s]', '', text)
        words = text.split()
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        return [word for word in words if len(word) > 2 and word not in stop_words]
    
    def fit(self, documents: List[str]):
        all_words = []
        doc_word_sets = []
        for doc in documents:
            words = self.preprocess_text(doc)
            all_words.extend(words)
            doc_word_sets.append(set(words))

        # Build vocabulary
        word_counts = Counter(all_words)
        self.vocabulary = {word: idx for idx, (word, count) in enumerate(word_counts.items()) if count >= 2}

        # Calculate IDF scores
        num_docs = len(documents)
        for word in self.vocabulary:
            doc_freq = sum(1 for doc_words in doc_word_sets if word in doc_words)
            self.idf_scores[word] = math.log(num_docs/doc_freq)
        
        self.is_fitted = True

    def embed(self, text:str) -> np.ndarray:
         #Convert text to embedding vector
        if not self.is_fitted:
            raise ValueError("Embedder not fitted, call fit() first.")
        
        words = self.preprocess_text(text)
        word_counts = Counter(words)
        #tf-idf xu ly nngu tu nhien
        vector = np.zeros(len(self.vocabulary))
        total_words = len(words)

        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / total_words
                idf = self.idf_scores[word]
                vector[self_vocabulary[word]] = tf * idf

        #Chuẩn hóa vector
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
class VectorStore:

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: np.darray = None

    def add_documents(self, docs: Document):
        self.documents.extend(docs)
        self._update_embeddings()

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        self._update_embeddings()

    def _update_embeddings(self):
        #update embeddings matrix
        if self.documents:
            self.embeddings = np.array([doc.embedding for doc in self.documents])
            if embeddings:
                self.embeddings = np.vstack(embeddings)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[Document, float]]:
        #find most similar documents
        if self.embeddings is None or len(self.documents) == 0:
            return []
        #tinh toan do tuong dong cosine
        similarities = np.dot(self.embeddings, query_embedding)

        #top k 
        results = []
        for i, sim in enumerate(similarities):
            if sim > threshold:
                results.append((self.documents[i], float(sim)))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
    def clear(self):
        self.documents = []
        self.embeddings = None

class SimpleRAG:
    def __init__(self):
         self.embedder = SimpleEmbedder()
         self.vector_store = VectorStore()
         self.is_ready = False

    def add_documents(self, docs: List[Document]):
        #Add a single document
        full_texts = [f"{doc['title']} {doc['content']}" for doc in documents]   

        self._fit_embedder(full_texts) 

        docs = []
        for i, doc_data in enumerate(documents):
            embedding = self.embedder.embed(full_texts[i])
            doc = Document(
                id = doc_data['id'],
                title = doc['Title'],
                content = doc_data['content'],
                embedding = embedding
            )
            docs.append(doc)
        
        self.vector_store.add_documents(docs)
    def __fit_embedder(self, texts: List[str]):
        self.embedder.fit(texts)
        self.is_ready = True

    def query(self, question: str, top_k: int = 3, similarity_threshold: float = 0.1) -> Dict:
        if not self.is_ready:
            return {
                "answer": "No documents have been added to the knowledge base yet.",
                "retrieved_docs": [],
                "query": question
            }
        
        query_embedding = self.embedder.embed(question)
        retrived_docs = self.vector_store.similarity_search(query_embedding, top_k=top_k, threshold=similarity_threshold)
        answer = self._generate_response(question, retrived_docs)
        return {
            "answer": answer,
            "retrieved docs": [
                {
                    "title": doc.title,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "similarity": similarity,
                    "id": doc.id
                }
                for doc, similarity in retrived_docs
            ],
            "query": question,
            "num_retrieved": len(retrived_docs)
        }
    def _generate_response(self, question: str, retrived_docs: List[Tuple[Document, float]]) -> str:
        if not retrived_docs:
            return "I'm sorry, I couldn't find any relevant information in the knowledge base."
        
        response_parts = [f"Base on the available documents: here are some relevant information '{query}':\n"]
        
        
def main():
    rag = SimpleRAG()
    sample_docs = []


if __name__ == "__main__":
    main()
        
