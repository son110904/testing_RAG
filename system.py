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
        self.vocabulary = {word: idx for idx, (word, _) in enumerate(word_counts.items()) if count >= 2}

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

