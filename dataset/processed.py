import os
import pickle
import nltk
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

nltk.download("punkt_tab", quiet=True)
nltk.download("punkt",     quiet=True)
nltk.download("wordnet",   quiet=True)
nltk.download("stopwords", quiet=True)

# =====================================================
# OUTPUT FOLDERS
# =====================================================
os.makedirs("processed/bow_vectors",   exist_ok=True)
os.makedirs("processed/tfidf_vectors", exist_ok=True)

# =====================================================
# STOPWORDS
# =====================================================
stop_words = set(stopwords.words("english"))
stop_words.update(stopwords.words("french"))
stop_words.update({
    "would", "could", "also", "one", "know", "get",
    "like", "thank", "im", "ive", "fig", "figure",
    "table", "al", "pp", "vol", "et", "ibid", "ref",
    "dun", "dune", "comm", "cett", "peut", "tout",
    "tous", "etre", "dont", "cela", "cette", "celui",
    "celle", "ceux", "quand", "bien", "alors", "ainsi",
    "donc", "mais", "avoir", "faire", "tres", "encore",
    "toute", "apres", "avant", "entre", "selon", "lors",
    "depuis", "nbsp", "amp", "quot",
})

# =====================================================
# STEMMERS
# =====================================================
stemmer_en = SnowballStemmer("english")
stemmer_fr = SnowballStemmer("french")

FRENCH_INDICATORS = {
    "le", "la", "les", "de", "du", "des",
    "un", "une", "est", "sont", "avec",
    "pour", "dans", "sur"
}

def stem_word(word):
    if word in FRENCH_INDICATORS:
        return stemmer_fr.stem(word)
    return stemmer_en.stem(word)

# =====================================================
# PREPROCESSING (même pipeline pour BoW et TF-IDF)
# =====================================================
def preprocess(text):
    words = word_tokenize(text.lower())
    words = [
        w for w in words
        if w.isalpha()
        and w not in stop_words
        and 2 < len(w) < 25
    ]
    stems = [stem_word(w) for w in words]
    return " ".join(stems)

# =====================================================
# LOAD DOCUMENTS
# =====================================================
input_folder = "processed/clean_texts"
documents  = []
filenames  = []

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        file_path = os.path.join(input_folder, filename)
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            documents.append(text)
            filenames.append(filename)
        except Exception as e:
            print(f"Erreur avec {filename}: {e}")

print(f"Nombre de documents chargés : {len(documents)}")

# =====================================================
# PREPROCESSING
# =====================================================
print("\n⏳ Prétraitement en cours...")
processed_documents = [preprocess(doc) for doc in documents]
print("✅ Prétraitement terminé")

# Sauvegarde
with open("processed/filenames.pkl",            "wb") as f: pickle.dump(filenames, f)
with open("processed/processed_documents.pkl",  "wb") as f: pickle.dump(processed_documents, f)

# =====================================================
# BAG OF WORDS
# =====================================================
print("\n" + "=" * 50)
print("BAG OF WORDS")
print("=" * 50)

bow_vectorizer = CountVectorizer(min_df=2, max_df=0.8)
bow_matrix     = bow_vectorizer.fit_transform(processed_documents)

print(f"Matrix Shape : {bow_matrix.shape}")
density = bow_matrix.nnz / (bow_matrix.shape[0] * bow_matrix.shape[1])
print(f"Density      : {density:.6f}")

bow_sum = bow_matrix.sum(axis=0)
bow_top = sorted(
    [(word, bow_sum[0, idx]) for word, idx in bow_vectorizer.vocabulary_.items()],
    key=lambda x: x[1], reverse=True
)
print("\nTop 20 BoW words:\n")
for word, score in bow_top[:20]:
    print(f"  {word:<20} {int(score)}")

# Sauvegarde BoW
with open("processed/bow_vectors/bow_matrix.pkl",     "wb") as f: pickle.dump(bow_matrix, f)
with open("processed/bow_vectors/bow_vectorizer.pkl", "wb") as f: pickle.dump(bow_vectorizer, f)
with open("processed/bow_vectors/bow_features.txt",    "w") as f:
    f.write("\n".join(bow_vectorizer.get_feature_names_out()))

print("\n✅ BoW sauvegardé")

# =====================================================
# TF-IDF
# =====================================================
print("\n" + "=" * 50)
print("TF-IDF")
print("=" * 50)

tfidf_vectorizer = TfidfVectorizer(min_df=2, max_df=0.8)
tfidf_matrix     = tfidf_vectorizer.fit_transform(processed_documents)

print(f"Matrix Shape : {tfidf_matrix.shape}")
density = tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])
print(f"Density      : {density:.6f}")

tfidf_sum = tfidf_matrix.sum(axis=0)
tfidf_top = sorted(
    [(word, tfidf_sum[0, idx]) for word, idx in tfidf_vectorizer.vocabulary_.items()],
    key=lambda x: x[1], reverse=True
)
print("\nTop 20 TF-IDF words:\n")
for word, score in tfidf_top[:20]:
    print(f"  {word:<20} {score:.2f}")

# Sauvegarde TF-IDF
with open("processed/tfidf_vectors/tfidf_matrix.pkl",     "wb") as f: pickle.dump(tfidf_matrix, f)
with open("processed/tfidf_vectors/tfidf_vectorizer.pkl", "wb") as f: pickle.dump(tfidf_vectorizer, f)
with open("processed/tfidf_vectors/tfidf_features.txt",    "w") as f:
    f.write("\n".join(tfidf_vectorizer.get_feature_names_out()))

print("\n✅ TF-IDF sauvegardé")

# =====================================================
# SUMMARY
# =====================================================
print("\n" + "=" * 50)
print("SUMMARY")
print("=" * 50)
print(f"Documents      : {len(documents)}")
print(f"Vocabulaire    : {bow_matrix.shape[1]} mots uniques")
print(f"BoW  Shape     : {bow_matrix.shape}")
print(f"TF-IDF Shape   : {tfidf_matrix.shape}")
print(f"\nAperçu doc 0 (brut)      : {documents[0][:200]}")
print(f"Aperçu doc 0 (prétraité) : {processed_documents[0][:200]}")
print("\n✅ Pipeline terminé")