"""
Script : Détection de la langue des PDFs (français / anglais / autre)
Librairie : langdetect
"""

import os
import pickle
from langdetect import detect, DetectorFactory
from langdetect.lang_detect_exception import LangDetectException
from collections import Counter

# Résultats reproductibles
DetectorFactory.seed = 42

# ─────────────────────────────────────────────
# Chargement des documents
# ─────────────────────────────────────────────
input_folder = "processed/clean_texts"

filenames = []
documents = []

for filename in sorted(os.listdir(input_folder)):
    if filename.endswith(".txt"):
        path = os.path.join(input_folder, filename)
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
        filenames.append(filename)
        documents.append(text)

print(f"📂 {len(documents)} documents chargés\n")

# ─────────────────────────────────────────────
# Détection de langue
# ─────────────────────────────────────────────
results = []

for filename, text in zip(filenames, documents):
    # Utiliser les 1000 premiers caractères (suffisant pour détecter)
    sample = text[:1000].strip()

    try:
        if len(sample) < 20:
            lang = "unknown"
        else:
            lang = detect(sample)
    except LangDetectException:
        lang = "unknown"

    # Simplifier : garder fr, en, autre
    if lang == "fr":
        category = "français"
    elif lang == "en":
        category = "anglais"
    else:
        category = f"autre ({lang})"

    results.append({
        "filename": filename,
        "lang":     lang,
        "category": category,
    })

# ─────────────────────────────────────────────
# Affichage des résultats
# ─────────────────────────────────────────────
print("=" * 60)
print("RÉSULTATS PAR LANGUE")
print("=" * 60)

fr_docs  = [r for r in results if r["lang"] == "fr"]
en_docs  = [r for r in results if r["lang"] == "en"]
oth_docs = [r for r in results if r["lang"] not in ("fr", "en")]

print(f"\n🇫🇷 FRANÇAIS ({len(fr_docs)} documents) :")
print("─" * 50)
for r in fr_docs:
    print(f"  {r['filename'][:60]}")

print(f"\n🇬🇧 ANGLAIS ({len(en_docs)} documents) :")
print("─" * 50)
for r in en_docs:
    print(f"  {r['filename'][:60]}")

if oth_docs:
    print(f"\n🌍 AUTRES LANGUES ({len(oth_docs)} documents) :")
    print("─" * 50)
    for r in oth_docs:
        print(f"  [{r['lang']}] {r['filename'][:55]}")

# ─────────────────────────────────────────────
# Résumé statistique
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("RÉSUMÉ")
print("=" * 60)
total = len(results)
print(f"  Total     : {total} documents")
print(f"  Français  : {len(fr_docs):>3} ({len(fr_docs)/total*100:.1f}%)")
print(f"  Anglais   : {len(en_docs):>3} ({len(en_docs)/total*100:.1f}%)")
if oth_docs:
    lang_counter = Counter(r["lang"] for r in oth_docs)
    for lang, count in lang_counter.most_common():
        print(f"  {lang:<10}: {count:>3} ({count/total*100:.1f}%)")

# ─────────────────────────────────────────────
# Sauvegarde
# ─────────────────────────────────────────────
with open("processed/language_detection.pkl", "wb") as f:
    pickle.dump(results, f)

# Fichier CSV lisible
with open("processed/language_detection.csv", "w", encoding="utf-8") as f:
    f.write("filename,langue,categorie\n")
    for r in results:
        f.write(f"{r['filename']},{r['lang']},{r['category']}\n")

print("\n✅ Résultats sauvegardés :")
print("   processed/language_detection.pkl")
print("   processed/language_detection.csv")