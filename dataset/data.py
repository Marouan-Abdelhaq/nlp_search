"""
Script : Télécharger des PDFs depuis arXiv
Thèmes : IA, Machine Learning, Data Science, Programmation, Réseaux
"""

import os
import time
import requests
import xml.etree.ElementTree as ET

# ─────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────
OUTPUT_DIR = "raw/pdf"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Nombre de PDFs par thème
MAX_PER_TOPIC = 20  # ← changez ce nombre selon vos besoins

# Thèmes de recherche arXiv
TOPICS = [
    "machine learning introduction",
    "deep learning neural networks",
    "artificial intelligence survey",
    "data science methodology",
    "computer networks protocol",
    "natural language processing",
    "python programming tutorial",
    "data preprocessing techniques",
    "information retrieval search",
    "reinforcement learning",
]

# ─────────────────────────────────────────────
# Fonction : rechercher sur arXiv
# ─────────────────────────────────────────────
def search_arxiv(query, max_results=20):
    url = "http://export.arxiv.org/api/query"
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }
    response = requests.get(url, params=params)
    root = ET.fromstring(response.text)
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    papers = []
    for entry in root.findall("atom:entry", ns):
        title = entry.find("atom:title", ns).text.strip().replace("\n", " ")
        arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
        papers.append({"title": title, "id": arxiv_id})
    return papers

# ─────────────────────────────────────────────
# Fonction : télécharger un PDF
# ─────────────────────────────────────────────
def download_pdf(arxiv_id, title, output_dir):
    # Nom de fichier propre
    safe_title = "".join(c for c in title[:50] if c.isalnum() or c in " _-").strip()
    filename = f"{safe_title}_{arxiv_id}.pdf"
    filepath = os.path.join(output_dir, filename)

    # Ne pas re-télécharger si déjà présent
    if os.path.exists(filepath):
        print(f"  ⏭️  Déjà téléchargé : {filename}")
        return True

    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"
    try:
        response = requests.get(pdf_url, timeout=30, stream=True)
        if response.status_code == 200:
            with open(filepath, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"  ✅ {filename}")
            return True
        else:
            print(f"  ❌ Erreur {response.status_code} : {arxiv_id}")
            return False
    except Exception as e:
        print(f"  ❌ Échec : {arxiv_id} → {e}")
        return False

# ─────────────────────────────────────────────
# Téléchargement principal
# ─────────────────────────────────────────────
total_downloaded = 0
already_seen = set()

for topic in TOPICS:
    print(f"\n📚 Thème : '{topic}'")
    papers = search_arxiv(topic, max_results=MAX_PER_TOPIC)
    print(f"   {len(papers)} articles trouvés")

    for paper in papers:
        arxiv_id = paper["id"]

        # Éviter les doublons entre thèmes
        if arxiv_id in already_seen:
            continue
        already_seen.add(arxiv_id)

        success = download_pdf(arxiv_id, paper["title"], OUTPUT_DIR)
        if success:
            total_downloaded += 1

        # Pause pour ne pas surcharger arXiv
        time.sleep(1.5)

# ─────────────────────────────────────────────
# Résumé final
# ─────────────────────────────────────────────
print(f"\n{'─'*50}")
print(f"✅ Total téléchargé : {total_downloaded} PDFs")
print(f"📁 Dossier : {OUTPUT_DIR}/")
print(f"{'─'*50}")