import os
import re
from collections import Counter

import matplotlib.pyplot as plt
from wordcloud import WordCloud

# =========================================================
# PATHS
# =========================================================

DATASET_FOLDER = "processed/clean_texts"

# =========================================================
# VARIABLES
# =========================================================

documents = []
document_lengths = []
all_text = ""

# =========================================================
# LOAD DOCUMENTS
# =========================================================

for filename in os.listdir(DATASET_FOLDER):

    if filename.endswith(".txt"):

        file_path = os.path.join(DATASET_FOLDER, filename)

        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:

                text = f.read()

                # remove extra spaces
                text = re.sub(r"\s+", " ", text)

                documents.append(text)

        except Exception as e:
            print(f"Error reading {filename}: {e}")

# =========================================================
# BASIC STATISTICS
# =========================================================

num_documents = len(documents)

for doc in documents:

    words = doc.split()

    document_lengths.append(len(words))

    all_text += " " + doc

all_words = all_text.lower().split()

total_words = len(all_words)

unique_words = len(set(all_words))

average_length = sum(document_lengths) / len(document_lengths)

longest_document = max(document_lengths)

shortest_document = min(document_lengths)

# =========================================================
# MOST FREQUENT WORDS
# =========================================================

word_freq = Counter(all_words)

most_common_words = word_freq.most_common(20)

# =========================================================
# PRINT RESULTS
# =========================================================

print("=" * 50)
print("DATASET STATISTICS")
print("=" * 50)

print(f"Number of documents : {num_documents}")
print(f"Total words         : {total_words}")
print(f"Unique words        : {unique_words}")

print(f"Average doc length  : {average_length:.2f} words")
print(f"Longest document    : {longest_document} words")
print(f"Shortest document   : {shortest_document} words")

print("\nTop 20 Most Frequent Words:\n")

for word, freq in most_common_words:
    print(f"{word:<20} {freq}")

# =========================================================
# HISTOGRAM - DOCUMENT LENGTHS
# =========================================================

plt.figure(figsize=(10, 5))

plt.hist(document_lengths, bins=30)

plt.xlabel("Document Length (words)")
plt.ylabel("Number of Documents")

plt.title("Distribution of Document Lengths")

plt.show()

# =========================================================
# BAR CHART - TOP WORDS
# =========================================================

top_words = [word for word, freq in most_common_words]
top_freqs = [freq for word, freq in most_common_words]

plt.figure(figsize=(12, 5))

plt.bar(top_words, top_freqs)

plt.xticks(rotation=45)

plt.xlabel("Words")
plt.ylabel("Frequency")

plt.title("Top 20 Most Frequent Words")

plt.show()

# =========================================================
# WORD CLOUD
# =========================================================

wordcloud = WordCloud(
    width=1200,
    height=600,
    background_color="white"
).generate(all_text)

plt.figure(figsize=(15, 7))

plt.imshow(wordcloud)

plt.axis("off")

plt.title("Word Cloud")

plt.show()