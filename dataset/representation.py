import os
import pickle
import numpy as np

from gensim.models import Word2Vec

os.makedirs("processed/w2v_vectors", exist_ok=True)

print("⏳ Chargement des documents...")

with open("processed/preprocessed_data/tokenized_documents.pkl", "rb") as f:
    tokenized_docs = pickle.load(f)

print(f"✅ {len(tokenized_docs)} documents chargés")

print("\n⏳ Entraînement Word2Vec...")

w2v_model = Word2Vec(
    sentences=tokenized_docs,
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    sg=1,
    epochs=10,
)

print(f"✅ Vocabulaire : {len(w2v_model.wv)} mots")

def document_vector(tokens, model):

    valid_words = [
        w for w in tokens
        if w in model.wv
    ]

    if not valid_words:
        return np.zeros(model.vector_size)

    vectors = [
        model.wv[w]
        for w in valid_words
    ]

    return np.mean(vectors, axis=0)

print("\n⏳ Création des vecteurs documents...")

w2v_matrix = np.array([
    document_vector(tokens, w2v_model)
    for tokens in tokenized_docs
])

print(f"✅ Matrice : {w2v_matrix.shape}")

w2v_model.save(
    "processed/w2v_vectors/word2vec.model"
)

with open(
    "processed/w2v_vectors/w2v_matrix.pkl",
    "wb"
) as f:
    pickle.dump(w2v_matrix, f)

print("\n✅ Word2Vec sauvegardé")

print("\n" + "="*50)
print("SUMMARY")
print("="*50)

print(f"Documents : {len(tokenized_docs)}")
print(f"Shape     : {w2v_matrix.shape}")

print("\n✅ Représentation terminée")