import os
import pickle
import nltk

from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)

os.makedirs("processed", exist_ok=True)

stop_words = set(stopwords.words("english"))
stop_words.update(stopwords.words("french"))

stop_words.update({
    "would", "could", "also", "one", "know",
    "like", "figure", "table", "nbsp", "quot"
})

stemmer_en = SnowballStemmer("english")
stemmer_fr = SnowballStemmer("french")

FRENCH_WORDS = {
    "le","la","les","de","du","des",
    "un","une","est","avec","pour"
}

def stem_word(word):

    if word in FRENCH_WORDS:
        return stemmer_fr.stem(word)

    return stemmer_en.stem(word)

def preprocess(text):

    words = word_tokenize(text.lower())

    words = [
        w for w in words
        if w.isalpha()
        and w not in stop_words
        and 2 < len(w) < 25
    ]

    stems = [stem_word(w) for w in words]

    return stems

input_folder = "processed/clean_texts"

documents = []
filenames = []

for filename in sorted(os.listdir(input_folder)):

    if filename.endswith(".txt"):

        path = os.path.join(input_folder, filename)

        try:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            documents.append(text)
            filenames.append(filename)

        except Exception as e:
            print(f"Erreur {filename}: {e}")

print(f"Documents chargés : {len(documents)}")

print("\n⏳ Prétraitement...")

tokenized_docs = [
    preprocess(doc)
    for doc in documents
]

string_docs = [
    " ".join(tokens)
    for tokens in tokenized_docs
]

print("✅ Prétraitement terminé")

with open("processed/preprocessed_data/processed_documents.pkl", "wb") as f:
    pickle.dump(string_docs, f)

with open("processed/preprocessed_data/tokenized_documents.pkl", "wb") as f:
    pickle.dump(tokenized_docs, f)

with open("processed/preprocessed_data/filenames.pkl", "wb") as f:
    pickle.dump(filenames, f)

print("\n✅ Documents sauvegardés")