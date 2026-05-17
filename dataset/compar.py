"""
Script : Comparaison BoW vs TF-IDF
Requêtes basées sur les VRAIS fichiers du dataset (FR + EN)
"""

import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords', quiet=True)
nltk.download('punkt',     quiet=True)
nltk.download('punkt_tab', quiet=True)

# ─────────────────────────────────────────────
# Chargement
# ─────────────────────────────────────────────
print("⏳ Chargement des modèles...")

with open("processed/bow_vectors/bow_vectorizer.pkl",   "rb") as f: bow_vectorizer   = pickle.load(f)
with open("processed/bow_vectors/bow_matrix.pkl",       "rb") as f: bow_matrix       = pickle.load(f)
with open("processed/tfidf_vectors/tfidf_vectorizer.pkl","rb") as f: tfidf_vectorizer = pickle.load(f)
with open("processed/tfidf_vectors/tfidf_matrix.pkl",   "rb") as f: tfidf_matrix     = pickle.load(f)
with open("processed/filenames.pkl",                    "rb") as f: filenames        = pickle.load(f)

print(f"✅ {len(filenames)} documents chargés")
print(f"   BoW    : {bow_matrix.shape}")
print(f"   TF-IDF : {tfidf_matrix.shape}\n")

# ─────────────────────────────────────────────
# Prétraitement des requêtes (identique à processed.py)
# ─────────────────────────────────────────────
stop_words = set(stopwords.words('english'))
stop_words.update(stopwords.words('french'))
stop_words.update({
    "would","could","also","one","know","get","like","thank","im","ive",
    "fig","figure","table","al","pp","vol","et","ibid","ref","dun","dune",
    "comm","cett","peut","tout","tous","etre","dont","cela","cette","celui",
    "celle","ceux","quand","bien","alors","ainsi","donc","mais","avoir",
    "faire","tres","encore","toute","apres","avant","entre","selon","lors",
    "depuis","nbsp","amp","quot",
})

stemmer_en = SnowballStemmer("english")
stemmer_fr = SnowballStemmer("french")
FRENCH_INDICATORS = {"le","la","les","de","du","des","un","une",
                     "est","sont","avec","pour","dans","sur"}

def stem_word(w):
    return stemmer_fr.stem(w) if w in FRENCH_INDICATORS else stemmer_en.stem(w)

def preprocess(query):
    words = word_tokenize(query.lower())
    words = [w for w in words if w.isalpha() and w not in stop_words and 2 < len(w) < 25]
    return " ".join([stem_word(w) for w in words])

# ─────────────────────────────────────────────
# Fonctions de recherche
# ─────────────────────────────────────────────
def search_bow(query, top_k=3):
    q      = preprocess(query)
    scores = cosine_similarity(bow_vectorizer.transform([q]), bow_matrix).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return [(filenames[i], round(float(scores[i]), 4)) for i in idx]

def search_tfidf(query, top_k=3):
    q      = preprocess(query)
    scores = cosine_similarity(tfidf_vectorizer.transform([q]), tfidf_matrix).flatten()
    idx    = scores.argsort()[::-1][:top_k]
    return [(filenames[i], round(float(scores[i]), 4)) for i in idx]

def short(name): return name.replace(".txt","")[:45]

# ─────────────────────────────────────────────
# Requêtes de test sur les VRAIS fichiers
# ─────────────────────────────────────────────
TEST_QUERIES = [
    # ── FRANÇAIS ──────────────────────────────────────────────
    (
        "introduction apprentissage automatique machine learning",
        ["introduction au machine learning", "introduction à l'apprentissage",
         "coursmachiinelearningia"],
        "FR"
    ),
    (
        "programmation Python débutant",
        ["cours de python", "apprendre à programmer avec python",
         "programmation avec python", "un zeste de python"],
        "FR"
    ),
    (
        "intelligence artificielle résolution problèmes",
        ["intelligence artificielle - résolution", "intelligence artificielle",
         "introduction à l'intelligence artificielle"],
        "FR"
    ),
    (
        "deep learning introduction réseau neuronal",
        ["introduction au deep learning", "introduction au machine learning",
         "introduction à l'apprentissage"],
        "FR"
    ),
    (
        "initiation informatique débutant",
        ["initiation à l'informatique", "introduction à l'informatique",
         "l'informatique pour débutants", "guide de formation"],
        "FR"
    ),
    (
        "algorithmique programmation orientée objet java",
        ["introduction à l'algorithmique", "les bases de la programmation",
         "introduction à java", "poo et java"],
        "FR"
    ),
    # ── ANGLAIS ───────────────────────────────────────────────
    (
        "machine learning optimization supervised",
        ["1706.10207", "dome recommendations", "a benchmark study of machine learning",
         "automatic machine learning"],
        "EN"
    ),
    (
        "neural network deep learning approximation",
        ["the modern mathematics of deep learning", "on the approximation",
         "dual accuracy-quality-driven neural"],
        "EN"
    ),
    (
        "reinforcement learning policy reward",
        ["a tutorial on meta-reinforcement", "directed policy gradient",
         "compression and localization in reinforcement", "anderson acceleration"],
        "EN"
    ),
    (
        "data science preprocessing methods challenges",
        ["comparative analysis of data preprocessing", "data science methodologies",
         "ambitious data science", "why data science projects fail"],
        "EN"
    ),
    (
        "natural language processing NLP text review",
        ["a comprehensive review of state-of-the-art",
         "exploring the landscape of natural language",
         "an open natural language processing"],
        "EN"
    ),
    (
        "network protocol communication security",
        ["a comparative review of internet protocol",
         "dtn7 an open-source disruption",
         "changing neighbors k secure sum protocol"],
        "EN"
    ),
    # ── ROBUSTESSE ────────────────────────────────────────────
    (
        "apprendre automatiquement exemples données",
        ["introduction à l'apprentissage", "introduction au machine learning",
         "coursmachiinelearningia"],
        "FR-approx"
    ),
    (
        "how machines learn from data examples",
        ["1805.05052", "a benchmark study", "1709.02840"],
        "EN-approx"
    ),
    (
        "rézeau de neuronne profond",
        ["introduction au deep learning", "the modern mathematics of deep learning",
         "memristors"],
        "FR-typo"
    ),
    (
        "comment fonctionne internet protocole réseau",
        ["a comparative review of internet protocol", "dtn7",
         "architecture des systèmes informatiques",
         "adaptation of turn protocol"],
        "FR-approx"
    ),
]

# ─────────────────────────────────────────────
# Évaluation
# ─────────────────────────────────────────────
print("=" * 70)
print("COMPARAISON BoW vs TF-IDF — Requêtes sur vrais fichiers")
print("=" * 70)

hits_bow   = []
hits_tfidf = []

for query, expected_docs, lang in TEST_QUERIES:
    res_b = search_bow(query,   top_k=3)
    res_t = search_tfidf(query, top_k=3)

    top3_b = " ".join([r[0].lower() for r in res_b])
    top3_t = " ".join([r[0].lower() for r in res_t])

    hit_b = any(exp[:20] in top3_b for exp in expected_docs)
    hit_t = any(exp[:20] in top3_t for exp in expected_docs)

    hits_bow.append(hit_b)
    hits_tfidf.append(hit_t)

    print(f"\n[{lang}] Requête : '{query}'")
    print(f"  Attendu  : {expected_docs[0][:55]}")
    print(f"  {'BoW':<10} {'TF-IDF'}")
    for i in range(3):
        nb, sb = res_b[i]
        nt, st = res_t[i]
        print(f"  {i+1}. {short(nb):<46} {short(nt)}")
        print(f"     score={sb:<42} score={st}")
    print(f"  Pertinent? BoW: {'✅' if hit_b else '❌'}  |  TF-IDF: {'✅' if hit_t else '❌'}")

# ─────────────────────────────────────────────
# Résultats finaux
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("RÉSULTATS FINAUX")
print("=" * 70)

def precision(hits): return sum(hits) / len(hits) if hits else 0
idx_fr     = [i for i,(_,_,l) in enumerate(TEST_QUERIES) if l == "FR"]
idx_en     = [i for i,(_,_,l) in enumerate(TEST_QUERIES) if l == "EN"]
idx_approx = [i for i,(_,_,l) in enumerate(TEST_QUERIES) if "approx" in l or "typo" in l]
def p(hits, idx): return precision([hits[i] for i in idx])

print(f"\n{'Catégorie':<30} {'BoW':>10} {'TF-IDF':>10}")
print("─" * 52)
print(f"{'Requêtes FR exactes (6)':<30} {p(hits_bow,idx_fr):>9.0%} {p(hits_tfidf,idx_fr):>9.0%}")
print(f"{'Requêtes EN exactes (6)':<30} {p(hits_bow,idx_en):>9.0%} {p(hits_tfidf,idx_en):>9.0%}")
print(f"{'Requêtes approximatives (4)':<30} {p(hits_bow,idx_approx):>9.0%} {p(hits_tfidf,idx_approx):>9.0%}")
print("─" * 52)
print(f"{'TOTAL Precision@3 (16)':<30} {precision(hits_bow):>9.0%} {precision(hits_tfidf):>9.0%}")

pb = precision(hits_bow)
pt = precision(hits_tfidf)
print()
if pt > pb:
    print(f"🏆 GAGNANT : TF-IDF  (+{pt-pb:.0%} de précision)")
    print("   → TF-IDF est la méthode principale recommandée")
elif pb > pt:
    print(f"🏆 GAGNANT : BoW  (+{pb-pt:.0%} de précision)")
else:
    print("🤝 ÉGALITÉ — préférer TF-IDF (meilleure discrimination)")
print("=" * 70)