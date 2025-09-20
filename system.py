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
    content: str  
    embedding: np.ndarray = None

class SimpleEmbedder:

    def __init__(self):
        self.vocabulary = {}
        self.idf_scores = {}
        self.is_fitted = False

    def preprocess_text(self, text: str) -> List[str]:  
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
        # Filter words with count >= 2 
        filtered_words = [word for word, count in word_counts.items() if count >= 2]
        self.vocabulary = {word: idx for idx, word in enumerate(filtered_words)}

        # Calculate IDF scores
        num_docs = len(documents)
        for word in self.vocabulary:
            doc_freq = sum(1 for doc_words in doc_word_sets if word in doc_words)
            self.idf_scores[word] = math.log(num_docs/doc_freq)
        
        self.is_fitted = True

    def embed(self, text: str) -> np.ndarray:
        # convert text -> embedding vector
        if not self.is_fitted:
            raise ValueError("Embedder not fitted, call fit() first.")
        
        words = self.preprocess_text(text)  # Fixed: was missing 'preprocess_text'
        word_counts = Counter(words)
        # tf-idf xu ly nngu tu nhien
        vector = np.zeros(len(self.vocabulary))
        total_words = len(words)

        for word, count in word_counts.items():
            if word in self.vocabulary:
                tf = count / total_words
                idf = self.idf_scores[word]
                vector[self.vocabulary[word]] = tf * idf  # Fixed: was 'self_vocabulary'

        # Chuẩn hóa vector
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 0 else vector
    
class VectorStore:

    def __init__(self):
        self.documents: List[Document] = []
        self.embeddings: np.ndarray = None  # Fixed: was 'np.darray'

    def add_document(self, doc: Document):  # Fixed: renamed from 'add_documents'
        self.documents.append(doc)  # Fixed: was 'extend'
        self._update_embeddings()

    def add_documents(self, docs: List[Document]):
        self.documents.extend(docs)
        self._update_embeddings()

    def _update_embeddings(self):
        # update embeddings matrix
        if self.documents:
            embeddings = [doc.embedding for doc in self.documents if doc.embedding is not None]  # Fixed: added variable declaration
            if embeddings:
                self.embeddings = np.vstack(embeddings)

    def similarity_search(self, query_embedding: np.ndarray, top_k: int = 5, threshold: float = 0.1) -> List[Tuple[Document, float]]:  # Fixed: added threshold parameter
        # find most similar documents
        if self.embeddings is None or len(self.documents) == 0:
            return []
        # tinh toan do tuong dong cosine
        similarities = np.dot(self.embeddings, query_embedding)

        # top k 
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

    def add_documents_batch(self, documents: List[Dict]):  # Fixed: renamed method and parameters
        # Add multiple documents at once
        full_texts = [f"{doc['title']} {doc['content']}" for doc in documents]   

        self._fit_embedder(full_texts) 

        docs = []
        for i, doc_data in enumerate(documents):
            embedding = self.embedder.embed(full_texts[i])
            doc = Document(
                id=doc_data['id'],  # Fixed: was missing '='
                title=doc_data['title'],  # Fixed: was 'doc['Title']'
                content=doc_data['content'],
                embedding=embedding
            )
            docs.append(doc)
        
        self.vector_store.add_documents(docs)
    
    def _fit_embedder(self, texts: List[str]):
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
        retrieved_docs = self.vector_store.similarity_search(query_embedding, top_k=top_k, threshold=similarity_threshold)  # Fixed: typo and parameter name
        answer = self._generate_response(question, retrieved_docs)
        return {
            "answer": answer,
            "retrieved_docs": [  # Fixed: key name
                {
                    "title": doc.title,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "similarity": similarity,
                    "id": doc.id
                }
                for doc, similarity in retrieved_docs  # Fixed: variable name
            ],
            "query": question,
            "num_retrieved": len(retrieved_docs)  # Fixed: variable name
        }
    
    def _generate_response(self, query: str, retrieved_docs: List[Tuple[Document, float]]) -> str:  # Fixed: parameter names
        if not retrieved_docs:
            return "I'm sorry, I couldn't find any relevant information in the knowledge base."
        
        response_parts = [f"Based on the available documents, here are some relevant information for '{query}':\n"]  # Fixed: typo
        
        for i, (doc, similarity) in enumerate(retrieved_docs, 1):
            confidence = "High" if similarity > 0.7 else "Medium" if similarity > 0.4 else "Low"
            response_parts.append(f"{i}. Title: {doc.title}\n   Content: {doc.content}\n   Similarity: {similarity:.4f} (Confidence: {confidence})\n")

        return "".join(response_parts)

    def get_stats(self) -> Dict:
        return {
            "num_documents": len(self.vector_store.documents),
            "vocabulary_size": len(self.embedder.vocabulary) if self.is_ready else 0,
            "is_ready": self.is_ready  # Fixed: key name
        }

    def clear_knowledge_base(self):
        self.vector_store.clear()
        self.embedder = SimpleEmbedder()
        self.is_ready = False
        
def main():
    rag = SimpleRAG()
    sample_docs = [
        {
            "id": "doc1",
            "title": "Python Programming",
            "content": "Python is a high-level programming language known for its simplicity and readability. It supports object-oriented, procedural, and functional programming paradigms. Python is widely used in web development, data science, machine learning, and automation. Key features include dynamic typing, automatic memory management, and extensive standard library."
        },
        {
            "id": "doc2", 
            "title": "Machine Learning Fundamentals",
            "content": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without explicit programming. Main types include supervised learning (classification and regression), unsupervised learning (clustering and dimensionality reduction), and reinforcement learning. Popular algorithms include linear regression, decision trees, random forests, and neural networks."
        },
        {
            "id": "doc3",
            "title": "Web Development with Flask",
            "content": "Flask is a lightweight web framework for Python. It's designed to make getting started with web development quick and easy, with the ability to scale up to complex applications. Flask provides tools, libraries and patterns to build web applications. Key components include routing, templates with Jinja2, and request handling."
        },
        {
            "id": "doc4",
            "title": "Data Science Tools",
            "content": "Data science combines statistics, programming, and domain knowledge to extract insights from data. Essential Python libraries include NumPy for numerical computing, Pandas for data manipulation, Matplotlib and Seaborn for visualization, and Scikit-learn for machine learning. Jupyter notebooks are commonly used for interactive data analysis."
        }
    ]
    print("🚀 Initializing Simple RAG System...")
    print("=" * 50)

    print("\n📚 Adding documents to knowledge base...")
    rag.add_documents_batch(sample_docs)

    stats = rag.get_stats()
    print(f"✅ Added {stats['num_documents']} documents")
    print(f"📖 Vocabulary size: {stats['vocabulary_size']} words")

    # Demo
    queries = [
        "What is Python?",
        "Tell me about machine learning",
        "How do I build web applications?", 
        "What tools are used in data science?",
        "Explain neural networks"  # This should have lower relevance
    ]

    print("\n" + "=" * 50)
    print("🔍 Testing Queries")
    print("=" * 50)

    for query in queries:
        print(f"\n❓ Query: {query}")  
        print("-" * 30)

        result = rag.query(query, top_k=2, similarity_threshold=0.1)
        print(f"📄 Retrieved {result['num_retrieved']} relevant documents")

        if result['retrieved_docs']:
            print("\n📋 Retrieved Documents:")
            for doc in result['retrieved_docs']:
                print(f"  • {doc['title']} (Similarity: {doc['similarity']:.3f})") 
        
        print(f"\n🤖 Generated Response:")
        print(result['answer'])
        print("\n" + "="*50)

if __name__ == "__main__":
    main()