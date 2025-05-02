import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np

# -------------------------------
# 1. CONFIG
# -------------------------------
SERPAPI_API_KEY = st.secrets["SERPAPI_API_KEY"]
SEARCH_ENGINE = "google"
MAX_SNIPPETS = 5

# -------------------------------
# 2. MODELS
# -------------------------------
st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-mpnet-base-v2")
    nli = pipeline("text-classification", model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    return embedder, nli

embedder, nli_pipeline = load_models()

# -------------------------------
# 3. UTILITIES
# -------------------------------
def extract_key_claims(text, num_sentences=2):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences])

def serpapi_search(query, num_results=5):
    params = {
        "engine": SEARCH_ENGINE,
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    response.raise_for_status()
    results = response.json()

    snippets = []
    if 'organic_results' in results:
        for result in results['organic_results'][:num_results]:
            snippet = result.get('snippet') or result.get('title') or ""
            snippets.append(snippet)
    return snippets

def verify_with_nli(claim, evidence):
    input_text = f"{claim} </s> {evidence}"
    result = nli_pipeline(input_text)[0]
    return result['label'], result['score']

# -------------------------------
# 4. FACT-CHECKING PIPELINE
# -------------------------------
def fact_check(article_text):
    claim = extract_key_claims(article_text)
    snippets = serpapi_search(claim, num_results=MAX_SNIPPETS)

    if not snippets:
        return "⚠️ No relevant results found. Cannot verify."

    filtered_snippets = []
    claim_embed = embedder.encode([claim])
    snippets_embed = embedder.encode(snippets)
    sims = cosine_similarity(claim_embed, snippets_embed)[0]

    for snippet, sim in zip(snippets, sims):
        if sim > 0.5:
            filtered_snippets.append(snippet)

    if not filtered_snippets:
        return "⚠️ No similar evidence found. Cannot verify."

    entail = contradict = 0
    for snippet in filtered_snippets:
        label, _ = verify_with_nli(claim, snippet)
        if label == "ENTAILMENT":
            entail += 1
        elif label == "CONTRADICTION":
            contradict += 1

    if entail > contradict:
        return "✅ Based on Fact"
    elif contradict > entail:
        return "❌ Likely False"
    else:
        return "⚠️ Possibly Misleading or Unclear"

# -------------------------------
# 5. STREAMLIT APP
# -------------------------------
st.set_page_config(page_title="Live Fact Checker", page_icon="✅")
st.title("🕵️ Live Fact Checking App")

article_input = st.text_area("Paste a news article or claim to verify:", height=250)

if st.button("Check Fact"):
    if not article_input.strip():
        st.warning("Please paste an article or claim.")
    else:
        with st.spinner("Fact-checking in progress..."):
            result = fact_check(article_input)
        st.markdown(f"### Result: {result}")
import streamlit as st
import requests
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import sent_tokenize
import numpy as np

# -------------------------------
# 1. CONFIG
# -------------------------------
SERPAPI_API_KEY = "YOUR_SERPAPI_KEY"
SEARCH_ENGINE = "google"
MAX_SNIPPETS = 5

# -------------------------------
# 2. MODELS
# -------------------------------
st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-mpnet-base-v2")
    nli = pipeline("text-classification", model="ynie/roberta-large-snli_mnli_fever_anli_R1_R2_R3-nli")
    return embedder, nli

embedder, nli_pipeline = load_models()

# -------------------------------
# 3. UTILITIES
# -------------------------------
def extract_key_claims(text, num_sentences=2):
    sentences = sent_tokenize(text)
    return " ".join(sentences[:num_sentences])

def serpapi_search(query, num_results=5):
    params = {
        "engine": SEARCH_ENGINE,
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "num": num_results,
    }
    response = requests.get("https://serpapi.com/search", params=params)
    response.raise_for_status()
    results = response.json()

    snippets = []
    if 'organic_results' in results:
        for result in results['organic_results'][:num_results]:
            snippet = result.get('snippet') or result.get('title') or ""
            snippets.append(snippet)
    return snippets

def verify_with_nli(claim, evidence):
    input_text = f"{claim} </s> {evidence}"
    result = nli_pipeline(input_text)[0]
    return result['label'], result['score']

# -------------------------------
# 4. FACT-CHECKING PIPELINE
# -------------------------------
def fact_check(article_text):
    claim = extract_key_claims(article_text)
    snippets = serpapi_search(claim, num_results=MAX_SNIPPETS)

    if not snippets:
        return "⚠️ No relevant results found. Cannot verify."

    filtered_snippets = []
    claim_embed = embedder.encode([claim])
    snippets_embed = embedder.encode(snippets)
    sims = cosine_similarity(claim_embed, snippets_embed)[0]

    for snippet, sim in zip(snippets, sims):
        if sim > 0.5:
            filtered_snippets.append(snippet)

    if not filtered_snippets:
        return "⚠️ No similar evidence found. Cannot verify."

    entail = contradict = 0
    for snippet in filtered_snippets:
        label, _ = verify_with_nli(claim, snippet)
        if label == "ENTAILMENT":
            entail += 1
        elif label == "CONTRADICTION":
            contradict += 1

    if entail > contradict:
        return "✅ Based on Fact"
    elif contradict > entail:
        return "❌ Likely False"
    else:
        return "⚠️ Possibly Misleading or Unclear"

# -------------------------------
# 5. STREAMLIT APP
# -------------------------------
st.set_page_config(page_title="Live Fact Checker", page_icon="✅")
st.title("🕵️ Live Fact Checking App")

article_input = st.text_area("Paste a news article or claim to verify:", height=250)

if st.button("Check Fact"):
    if not article_input.strip():
        st.warning("Please paste an article or claim.")
    else:
        with st.spinner("Fact-checking in progress..."):
            result = fact_check(article_input)
        st.markdown(f"### Result: {result}")
