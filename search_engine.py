import pickle
import math
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
class SearchEngine:
    def __init__(self):
        self.w2v_model = Word2Vec.load("dataset/processed/w2v_vectors/word2vec.model")
        with open("dataset/processed/w2v_vectors/w2v_matrix.pkl", "rb") as f:
            self.matrix = pickle.load(f)
        with open("dataset/processed/preprocessed_data/filenames.pkl", "rb") as f:
            self.filenames = pickle.load(f)
        with open("dataset/processed/preprocessed_data/processed_documents.pkl", "rb") as f:
            processed_docs = pickle.load(f)
        self.stop_words = set(stopwords.words('english')).union(stopwords.words('french'))
        self.stop_words.update({"would","could","also","one","know","get","like","thank","im","ive","fig","figure","table","al","pp","vol","et","ibid","ref","dun","dune","comm","cett","peut","tout","tous","etre","dont","cela","cette","celui","celle","ceux","quand","bien","alors","ainsi","donc","mais","avoir","faire","tres","encore","toute","apres","avant","entre","selon","lors","depuis","nbsp","amp","quot"})
        self.stemmer_en = SnowballStemmer("english")
        self.stemmer_fr = SnowballStemmer("french")
        self.french_indicators = {"le","la","les","de","du","des","un","une","est","sont","avec","pour","dans","sur"}
        self.bm25_docs = [doc.split() for doc in processed_docs]
        self.bm25_N = len(self.bm25_docs)
        self.bm25_avgdl = sum(len(d) for d in self.bm25_docs) / self.bm25_N
        self.bm25_df = {}
        for doc in self.bm25_docs:
            for word in set(doc):
                self.bm25_df[word] = self.bm25_df.get(word, 0) + 1
    def preprocess_query(self, query):
        words = word_tokenize(query.lower())
        is_french = any(w in self.french_indicators for w in words)
        clean_words = [w for w in words if w.isalpha() and w not in self.stop_words and 1 < len(w) < 25]
        if not is_french and hasattr(self, 'w2v_model'):
            vocab = self.w2v_model.wv
            fr_matches = sum(1 for w in clean_words if self.stemmer_fr.stem(w) in vocab)
            en_matches = sum(1 for w in clean_words if self.stemmer_en.stem(w) in vocab)
            if fr_matches > en_matches:
                is_french = True        
        stemmer = self.stemmer_fr if is_french else self.stemmer_en
        return " ".join([stemmer.stem(w) for w in clean_words])
    def get_query_vector(self, query):
        clean_query = self.preprocess_query(query)
        tokens = clean_query.split()
        valid_words = [word for word in tokens if word in self.w2v_model.wv]
        if not valid_words:
            return np.zeros((1, self.w2v_model.vector_size))
        word_vectors = np.array([self.w2v_model.wv[word] for word in valid_words])
        doc_vec = word_vectors.mean(axis=0)
        return doc_vec.reshape(1, -1)
    def search_cosine(self, query, top_k=5):
        query_vector = self.get_query_vector(query)
        if not query_vector.any():
            return []
        scores = cosine_similarity(query_vector, self.matrix).flatten()
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0: 
                results.append({
                    "document": self.filenames[idx],
                    "score": round(float(scores[idx]), 4),
                    "method": "Cosine"
                })
        return results
    def search_bm25(self, query, top_k=5):
        clean_query = self.preprocess_query(query)
        query_tokens = clean_query.split()
        if not query_tokens:
            return []
        k1 = 1.5
        b = 0.75
        scores = []
        for i, doc in enumerate(self.bm25_docs):
            score = 0.0
            doc_len = len(doc)
            tf_map = {}
            for w in doc:
                tf_map[w] = tf_map.get(w, 0) + 1
            for term in query_tokens:
                if term not in self.bm25_df:
                    continue
                df = self.bm25_df[term]
                idf = math.log((self.bm25_N - df + 0.5) / (df + 0.5) + 1)
                tf = tf_map.get(term, 0)
                numerator = tf * (k1 + 1)
                denominator = tf + k1 * (1 - b + b * doc_len / self.bm25_avgdl)
                score += idf * (numerator / denominator)
            scores.append(score)
        scores = np.array(scores)
        top_indices = scores.argsort()[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "document": self.filenames[idx],
                    "score": round(float(scores[idx]), 4),
                    "method": "BM25"
                })
        return results