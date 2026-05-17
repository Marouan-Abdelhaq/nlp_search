import os
import re

input_folder = "processed/extracted_texts"
output_folder = "processed/clean_texts"

os.makedirs(output_folder, exist_ok=True)

def clean_text(text):

    text = re.sub(r'\bcid\d+\w*\b', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'\S+@\S+', ' ', text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-zA-Z0-9àâäéèêëîïôùûüçÀÂÄÉÈÊËÎÏÔÙÛÜÇ\s]", "", text)
    words = text.split()
    words = [w for w in words if len(w) <= 25]
    text = text.strip()

    return text

saved = 0
skipped = 0

for root, dirs, files in os.walk(input_folder):

    for filename in files:

        if filename.endswith(".txt"):

            path = os.path.join(root, filename)

            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()

            cleaned = clean_text(text)

            if len(cleaned.split()) < 20:
                skipped += 1
                continue

            output_path = os.path.join(output_folder, filename)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned)
            saved += 1

print(f"✅ Nettoyage terminé : {saved} fichiers sauvegardés, {skipped} ignorés.")