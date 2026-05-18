import os
import re
import json
import time
from flask import Flask, request, jsonify, send_from_directory
from search_engine import SearchEngine
app = Flask(__name__, static_folder="static")
engine = SearchEngine()
metadata_path = os.path.join("metadata", "documents_metadata.json")
if os.path.exists(metadata_path):
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = {}
CLEAN_DIR = os.path.join("processed/preprocessed_data", "clean_texts")
def get_snippet(filename, query, max_len=220):
    txt_path = os.path.join(CLEAN_DIR, filename)
    if not os.path.exists(txt_path):
        return ""
    try:
        with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    except Exception:
        return ""
    if not text.strip():
        return ""
    query_words = [w.lower() for w in query.split() if len(w) > 2]
    text_lower = text.lower()
    best_pos = -1
    for word in query_words:
        pos = text_lower.find(word)
        if pos != -1:
            best_pos = pos
            break
    if best_pos == -1:
        snippet = text[:max_len].strip()
    else:
        start = max(0, best_pos - 80)
        end = min(len(text), best_pos + 140)
        snippet = text[start:end].strip()
        if start > 0:
            snippet = "... " + snippet
        if end < len(text):
            snippet = snippet + " ..."
    if len(snippet) > max_len:
        snippet = snippet[:max_len].rsplit(" ", 1)[0] + " ..."
    return snippet

@app.route("/")
def index():
    return send_from_directory("static", "index.html")
@app.route("/api/search")
def api_search():
    query = request.args.get("q", "").strip()
    algo = request.args.get("algo", "cosine").lower()
    top_k = request.args.get("k", "5")
    try:
        top_k = int(top_k)
        top_k = max(1, min(top_k, 20))
    except ValueError:
        top_k = 5
    if not query:
        return jsonify({"results": [], "time": 0, "query": ""})
    t0 = time.time()
    results_data = []
    if algo == "compare":
        res_cos = engine.search_cosine(query, top_k)
        res_bm25 = engine.search_bm25(query, top_k)
        elapsed = round(time.time() - t0, 4)
        for r in res_cos:
            doc = r["document"]
            meta = metadata.get(doc, {})
            results_data.append({
                "title": doc.replace(".txt", ""),
                "filename": doc,
                "score": r["score"],
                "method": "Cosine",
                "lang": meta.get("langue", ""),
                "theme": meta.get("theme", ""),
                "source": meta.get("source", ""),
                "snippet": get_snippet(doc, query),
            })
        bm25_data = []
        for r in res_bm25:
            doc = r["document"]
            meta = metadata.get(doc, {})
            bm25_data.append({
                "title": doc.replace(".txt", ""),
                "filename": doc,
                "score": r["score"],
                "method": "BM25",
                "lang": meta.get("langue", ""),
                "theme": meta.get("theme", ""),
                "source": meta.get("source", ""),
                "snippet": get_snippet(doc, query),
            })
        return jsonify({
            "results": results_data,
            "knn_results": bm25_data,
            "time": elapsed,
            "query": query,
            "algo": "compare",
        })
    else:
        if algo == "knn":
            raw = engine.search_bm25(query, top_k)
        else:
            raw = engine.search_cosine(query, top_k)
        elapsed = round(time.time() - t0, 4)
        for r in raw:
            doc = r["document"]
            meta = metadata.get(doc, {})
            results_data.append({
                "title": doc.replace(".txt", ""),
                "filename": doc,
                "score": r["score"],
                "method": r["method"],
                "lang": meta.get("langue", ""),
                "theme": meta.get("theme", ""),
                "source": meta.get("source", ""),
                "snippet": get_snippet(doc, query),
            })
        return jsonify({
            "results": results_data,
            "time": elapsed,
            "query": query,
            "algo": algo,
        })
if __name__ == "__main__":
    app.run(debug=False, port=5000)
