# 📚 Système de Recherche Documentaire — Partie 1 : Dataset & Représentation Textuelle

## 👤 Auteur
Étudiant : **Abdelhaq**
Partie : Collecte des données, Prétraitement, Représentation textuelle (BoW & TF-IDF)

---

## 🎯 Objectif de cette partie

Construire un **dataset de documents** (PDFs académiques) et les transformer en **représentations vectorielles** exploitables par un moteur de recherche documentaire.

Pipeline complet :
```
PDFs bruts → Extraction texte → Nettoyage → Prétraitement → BoW / TF-IDF → Comparaison
```

---

## 📁 Structure du projet

```
dataset/
├── raw/                        ← Documents bruts originaux
│   └── pdf/                    ← 249 fichiers PDF (cours + articles)
│
├── processed/                  ← Documents traités (généré automatiquement)
│   ├── extracted_texts/        ← Texte brut extrait des PDFs (.txt)
│   ├── clean_texts/            ← Texte nettoyé (.txt)
│   ├── bow_vectors/            ← Matrices BoW sérialisées
│   │   ├── bow_matrix.pkl      ← Matrice creuse (249 × vocab)
│   │   ├── bow_vectorizer.pkl  ← Vectorizer fitted (CountVectorizer)
│   │   └── bow_features.txt    ← Liste des mots du vocabulaire
│   ├── tfidf_vectors/          ← Matrices TF-IDF sérialisées
│   │   ├── tfidf_matrix.pkl    ← Matrice creuse (249 × vocab)
│   │   ├── tfidf_vectorizer.pkl← Vectorizer fitted (TfidfVectorizer)
│   │   └── tfidf_features.txt  ← Liste des mots du vocabulaire
│   ├── filenames.pkl           ← Liste ordonnée des noms de fichiers
│   ├── processed_documents.pkl ← Textes prétraités (stemmés)
│   ├── language_detection.pkl  ← Résultats détection de langue (dict)
│   └── language_detection.csv  ← Résultats détection de langue (CSV)
│
├── metadata/                   ← Métadonnées des documents
│   └── documents_metadata.json ← Fiche par document (langue, thème, taille...)
│
├── extract_pdf.py              ← Script 1 : extraction texte des PDFs
├── nettoyer_text.py            ← Script 2 : nettoyage du texte
├── processed.py                ← Script 3 : prétraitement + BoW + TF-IDF
├── detect.py                   ← Script 4 : détection de langue
├── metadata.py                 ← Script 5 : génération des métadonnées
├── compar.py                   ← Script 6 : comparaison BoW vs TF-IDF
├── eda.py                      ← Script 7 : analyse exploratoire (graphiques)
├── data.py                     ← Script 8 : téléchargement dataset TXT (newsgroups)
├── Figure_1.png                ← Distribution des longueurs de documents
├── Figure_2.png                ← Top 20 mots les plus fréquents
└── Figure_3.png                ← Word Cloud du corpus
```

---

## 📄 Description des scripts

### `extract_pdf.py` — Extraction du texte
Extrait le texte brut de chaque PDF dans `raw/pdf/` et le sauvegarde en `.txt` dans `processed/extracted_texts/`.

```bash
python3 extract_pdf.py
```

**Librairie utilisée :** `pdfplumber` ou `PyMuPDF`
**Entrée  :** `raw/pdf/*.pdf`
**Sortie  :** `processed/extracted_texts/*.txt`

---

### `nettoyer_text.py` — Nettoyage du texte
Nettoie les fichiers texte extraits : supprime les artefacts PDF (`cid12`), URLs, caractères spéciaux, mots trop longs (collés).

```bash
python3 nettoyer_text.py
```

**Entrée  :** `processed/extracted_texts/*.txt`
**Sortie  :** `processed/clean_texts/*.txt`

Opérations appliquées :
- Suppression artefacts PDF (`cid0`, `cid12`...)
- Suppression URLs et emails
- Conservation des lettres françaises accentuées (`é`, `à`, `ç`...)
- Suppression des mots > 25 caractères (mots collés)
- Ignoré si < 20 mots après nettoyage

---

### `processed.py` — Prétraitement + BoW + TF-IDF ⭐ Script principal
Script central du pipeline. Charge les textes nettoyés, applique le prétraitement, puis construit les deux représentations vectorielles.

```bash
python3 processed.py
```

**Entrée  :** `processed/clean_texts/*.txt`
**Sortie  :**
- `processed/filenames.pkl`
- `processed/processed_documents.pkl`
- `processed/bow_vectors/`
- `processed/tfidf_vectors/`

**Prétraitement appliqué :**
1. Tokenisation (`word_tokenize`)
2. Mise en minuscules
3. Suppression stopwords FR + EN (NLTK + liste custom)
4. Suppression mots courts (≤ 2 chars) et longs (≥ 25 chars)
5. Stemming bilingue (`SnowballStemmer` EN + FR)

**Paramètres des vectorizers :**
- `min_df=2` — ignore les mots qui apparaissent dans moins de 2 documents
- `max_df=0.8` — ignore les mots présents dans plus de 80% des documents

**Résultats obtenus :**
```
Documents      : 249
Vocabulaire    : 31 060 mots uniques
BoW  Shape     : (249, 31060)
TF-IDF Shape   : (249, 31060)
Densité        : ~4%
```

---

### `detect.py` — Détection de langue
Détecte automatiquement la langue de chaque document (français / anglais).

```bash
python3 detect.py
```

**Librairie :** `langdetect`
**Entrée  :** `processed/clean_texts/*.txt`
**Sortie  :** `processed/language_detection.pkl` et `.csv`

**Résultats :**
```
Français  :  61 documents (24.5%)
Anglais   : 188 documents (75.5%)
```

---

### `metadata.py` — Génération des métadonnées
Crée une fiche descriptive pour chaque document et la sauvegarde en JSON.

```bash
python3 metadata.py
```

**Entrée  :** `processed/filenames.pkl`, `processed/language_detection.pkl`
**Sortie  :** `metadata/documents_metadata.json`

**Contenu d'une fiche :**
```json
{
  "Cours de Python.txt": {
    "filename": "Cours de Python.txt",
    "langue": "fr",
    "theme": "Programmation",
    "nb_mots": 31450,
    "taille_txt_kb": 180.2,
    "taille_pdf_kb": 842.0,
    "source": "cours"
  }
}
```

**Thèmes détectés automatiquement :**
`Intelligence Artificielle`, `Machine Learning`, `Data Science`,
`Programmation`, `Réseaux`, `Information Retrieval`

---

### `compar.py` — Comparaison BoW vs TF-IDF ⭐
Évalue et compare les deux méthodes de représentation sur 16 requêtes de test couvrant les deux langues et différents niveaux de difficulté.

```bash
python3 compar.py
```

**Entrée  :** `processed/bow_vectors/`, `processed/tfidf_vectors/`, `processed/filenames.pkl`
**Métrique :** Precision@3 (le bon document est-il dans les 3 premiers résultats ?)

**Requêtes testées :**
- 6 requêtes françaises exactes
- 6 requêtes anglaises exactes
- 4 requêtes approximatives (synonymes, fautes de frappe, langage naturel)

**Résultats obtenus :**

| Catégorie | BoW | TF-IDF |
|---|---|---|
| Requêtes FR exactes | ~50% | ~67% |
| Requêtes EN exactes | ~67% | ~83% |
| Requêtes approximatives | ~25% | ~25% |
| **TOTAL** | **~50%** | **~63%** |

**Conclusion : TF-IDF est la méthode retenue** car elle pondère intelligemment les termes rares et discriminants, ce qui améliore la précision du ranking.

---

### `eda.py` — Analyse Exploratoire
Génère les graphiques statistiques du dataset.

```bash
python3 eda.py
```

**Sorties :**
- `Figure_1.png` — Distribution des longueurs de documents
- `Figure_2.png` — Top 20 mots les plus fréquents
- `Figure_3.png` — Word Cloud du corpus

---

### `data.py` — Téléchargement dataset TXT (optionnel)
Télécharge le dataset 20 Newsgroups et sauvegarde chaque article en `.txt`. Script utilisé en phase initiale, remplacé ensuite par les PDFs.

```bash
python3 data.py
```

---

## 🔑 Fichiers clés pour la partie suivante (moteur de recherche)

Votre partie a besoin de ces fichiers pour implémenter la recherche :

| Fichier | Contenu | Usage |
|---|---|---|
| `processed/tfidf_vectors/tfidf_matrix.pkl` | Matrice TF-IDF (249 × 31060) | Similarité cosinus avec la requête |
| `processed/tfidf_vectors/tfidf_vectorizer.pkl` | Vectorizer fitted | Transformer la requête utilisateur |
| `processed/bow_vectors/bow_matrix.pkl` | Matrice BoW (249 × 31060) | Comparaison avec TF-IDF |
| `processed/bow_vectors/bow_vectorizer.pkl` | Vectorizer fitted | Transformer la requête utilisateur |
| `processed/filenames.pkl` | Liste des 249 noms de fichiers | Retrouver le nom du document par son index |
| `metadata/documents_metadata.json` | Métadonnées (langue, thème...) | Afficher les infos du document retourné |

---

## 🚀 Comment reproduire le pipeline complet

```bash
# 1. Installer les dépendances
pip install pdfplumber scikit-learn nltk langdetect matplotlib wordcloud

# 2. Extraire le texte des PDFs
python3 extract_pdf.py

# 3. Nettoyer le texte
python3 nettoyer_text.py

# 4. Prétraitement + BoW + TF-IDF
python3 processed.py

# 5. Détection de langue
python3 detect.py

# 6. Générer les métadonnées
python3 metadata.py

# 7. Comparer les méthodes
python3 compar.py

# 8. Générer les graphiques (optionnel)
python3 eda.py
```

---

## 📦 Dépendances

```
scikit-learn
nltk
langdetect
pdfplumber
numpy
matplotlib
wordcloud
pickle (built-in)
```

---

## 📊 Statistiques du dataset

| Métrique | Valeur |
|---|---|
| Nombre de documents | 249 |
| Langue française | 61 (24.5%) |
| Langue anglaise | 188 (75.5%) |
| Total de mots | ~5 500 000 |
| Mots uniques (vocabulaire) | 31 060 |
| Longueur moyenne | ~733 mots/doc |
| Document le plus long | 187 988 mots |

---

## 💡 Choix technique justifié — BoW vs TF-IDF

**BoW (Bag of Words)** : compte le nombre brut d'occurrences de chaque mot. Simple mais favorise les longs documents et les mots fréquents sans distinction.

**TF-IDF** : pondère chaque mot par sa fréquence dans le document (`TF`) divisée par sa fréquence dans tout le corpus (`IDF`). Un mot rare dans le corpus mais fréquent dans un document reçoit un score élevé → meilleure discrimination entre documents.

**TF-IDF est retenu** comme méthode principale car il obtient une Precision@3 supérieure de ~13 points sur nos 16 requêtes de test, particulièrement sur les requêtes techniques en anglais.