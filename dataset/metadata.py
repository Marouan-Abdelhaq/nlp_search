"""
Script : Génération des métadonnées du dataset
Produit : metadata/documents_metadata.json
"""

import os
import json
import pickle

# ─────────────────────────────────────────────
# Chargement des infos déjà calculées
# ─────────────────────────────────────────────
with open("processed/filenames.pkl", "rb") as f:
    filenames = pickle.load(f)

# Détection de langue déjà faite
with open("processed/language_detection.pkl", "rb") as f:
    lang_results = pickle.load(f)

lang_map = {r["filename"]: r["lang"] for r in lang_results}

# ─────────────────────────────────────────────
# Détection automatique du thème selon le nom
# ─────────────────────────────────────────────
THEME_KEYWORDS = {
    "Intelligence Artificielle": [
        "intelligence artificielle", "artificial intelligence",
        "ia ", " ia-", "ai ", "chatgpt", "donner un sens"
    ],
    "Machine Learning": [
        "machine learning", "apprentissage automatique",
        "deep learning", "apprentissage", "reinforcement",
        "neural", "neuron", "1709", "1805", "2409", "2102",
        "2301", "2312", "1706", "coursmachiinelearning"
    ],
    "Data Science": [
        "data science", "data clustering", "preprocessing",
        "dataset", "data encoding", "data preproc",
        "ambitious data", "why data"
    ],
    "Programmation": [
        "python", "java", "programmation", "programming",
        "algorithmique", "algorithme", "langage", "c++",
        "web", "php", "poo", "objet", "logic program",
        "declarative", "prolog"
    ],
    "Réseaux": [
        "network", "réseau", "protocol", "internet",
        "communication", "dtn", "routing", "sip", "turn",
        "unicast", "broadcast", "peer"
    ],
    "Information Retrieval": [
        "information retrieval", "search", "retrieval",
        "bibliometric", "indexing"
    ],
}

def detect_theme(filename):
    name_lower = filename.lower()
    for theme, keywords in THEME_KEYWORDS.items():
        if any(kw in name_lower for kw in keywords):
            return theme
    return "Autre"

# ─────────────────────────────────────────────
# Construction des métadonnées
# ─────────────────────────────────────────────
os.makedirs("metadata", exist_ok=True)

metadata = {}
clean_folder    = "processed/clean_texts"
extracted_folder= "processed/extracted_texts"

for filename in filenames:
    # Taille du texte nettoyé
    clean_path = os.path.join(clean_folder, filename)
    nb_mots    = 0
    taille_kb  = 0

    if os.path.exists(clean_path):
        with open(clean_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        nb_mots   = len(text.split())
        taille_kb = round(os.path.getsize(clean_path) / 1024, 1)

    # Nom du PDF original
    pdf_name = filename.replace(".txt", ".pdf")
    pdf_path = os.path.join("raw/pdf", pdf_name)
    pdf_size = 0
    if os.path.exists(pdf_path):
        pdf_size = round(os.path.getsize(pdf_path) / 1024, 1)

    metadata[filename] = {
        "filename":     filename,
        "pdf_original": pdf_name,
        "langue":       lang_map.get(filename, "unknown"),
        "theme":        detect_theme(filename),
        "nb_mots":      nb_mots,
        "taille_txt_kb": taille_kb,
        "taille_pdf_kb": pdf_size,
        "source":       "arxiv" if filename[:4].isdigit() else "cours",
    }

# ─────────────────────────────────────────────
# Sauvegarde JSON
# ─────────────────────────────────────────────
output_path = "metadata/documents_metadata.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, ensure_ascii=False, indent=2)

print(f"✅ {len(metadata)} fiches créées → {output_path}")

# ─────────────────────────────────────────────
# Résumé par thème et langue
# ─────────────────────────────────────────────
from collections import Counter

themes = Counter(v["theme"]  for v in metadata.values())
langs  = Counter(v["langue"] for v in metadata.values())

print("\n📊 Répartition par thème :")
for theme, count in themes.most_common():
    print(f"  {theme:<30} {count:>3} documents")

print("\n🌍 Répartition par langue :")
for lang, count in langs.most_common():
    label = {"fr": "Français", "en": "Anglais"}.get(lang, lang)
    print(f"  {label:<15} {count:>3} documents")

print(f"\n📁 Fichier généré : {output_path}")