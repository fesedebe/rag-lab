# Build a small hybrid corpus by combining PubMed abstracts and FDA guideline snippets. 
# Saves .txt files to ./data/.

import os
import re
import time
import requests
from xml.etree import ElementTree
from bs4 import BeautifulSoup

# Configuration Defaults
PUBMED_DEFAULT_TERMS = [
    "clinical trial design",
    "eligibility criteria oncology",
    "oncology endpoints",
    "protocol optimization",
    "regulatory compliance AI"
]

FDA_GUIDELINES = {
    "good_clinical_practice": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e6r3-good-clinical-practice-gcp", 
    "statistical_principles": "https://www.fda.gov/regulatory-information/search-fda-guidance-documents/e9-statistical-principles-clinical-trials",
    "computerized_systems": "https://www.fda.gov/inspections-compliance-enforcement-and-criminal-investigations/fda-bioresearch-monitoring-information/guidance-industry-computerized-systems-used-clinical-trials"
}

OUT_DIR = "data"
NCBI_API_KEY = os.getenv("NCBI_API_KEY")

# PubMed Utilities
# Search PubMed and return article IDs for a given query
def fetch_pubmed_ids(query, n=2):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {"db": "pubmed", "term": query, "retmode": "xml", "retmax": n}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.text)
    return [id_tag.text for id_tag in root.findall(".//Id")]

# Fetch title and abstract text for a given PubMed ID
def fetch_abstract(pmid):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    params = {"db": "pubmed", "id": pmid, "retmode": "xml"}
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    root = ElementTree.fromstring(resp.text)
    title = root.findtext(".//ArticleTitle") or "Untitled"
    abstract = " ".join([t.text for t in root.findall(".//AbstractText") if t.text])
    return title.strip(), abstract.strip()

# Fetch abstracts for a list of PubMed search terms
def fetch_pubmed_corpus(terms, n_results=2, sleep=1.5, verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    for term in terms:
        if verbose:
            print(f"Querying PubMed: {term}")
        try:
            pmids = fetch_pubmed_ids(term, n=n_results)
        except requests.exceptions.HTTPError as e:
            if verbose:
                print(f"  Skipped '{term}' due to HTTP error: {e}")
            continue

        for i, pmid in enumerate(pmids, start=1):
            try:
                title, abstract = fetch_abstract(pmid)
                filename = f"pubmed_{term.replace(' ', '_')}_{i}.txt"
                save_text_file(filename, title, abstract)
                if verbose:
                    print(f"  Saved: {filename}")
            except Exception as e:
                if verbose:
                    print(f"  Skipped {pmid}: {e}")
            time.sleep(sleep)

# FDA Utilities
def clean_text(text):
    text = re.sub(r"\s+", " ", text)
    return text.replace("\xa0", " ").strip()

# Fetch a few paragraphs from an FDA guidance webpage
def fetch_fda_guideline_snippet(url, n_paragraphs=3):
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; CorpusFetcher/1.0)",
        "Accept-Language": "en-US,en;q=0.9"
    }
    resp = requests.get(url, headers=headers, timeout=10)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    paragraphs = soup.find_all("p")
    if not paragraphs:
        return "No content found."
    snippet = " ".join(clean_text(p.get_text()) for p in paragraphs[:n_paragraphs])
    return snippet

# Fetch text snippets from multiple FDA guidance documents
def fetch_fda_corpus(guidelines, verbose=True):
    os.makedirs(OUT_DIR, exist_ok=True)
    for name, url in guidelines.items():
        try:
            snippet = fetch_fda_guideline_snippet(url)
            filename = f"fda_{name}.txt"
            save_text_file(filename, name.replace('_', ' ').title(), snippet)
            if verbose:
                print(f"Saved: {filename}")
        except Exception as e:
            if verbose:
                print(f"Skipped {name}: {e}")

# Shared Utility - Write text content to ./data/filename.
def save_text_file(filename, title, body):
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, filename), "w", encoding="utf-8") as f:
        f.write(f"{title}\n\n{body}\n")

# Build PubMed & FDA corpora
def gather_biomed_corpus(
    pubmed_terms=None,
    n_results=2,
    sleep=1.5,
    include_fda=True,
    verbose=True
):
    if pubmed_terms:
        fetch_pubmed_corpus(pubmed_terms, n_results=n_results, sleep=sleep, verbose=verbose)
    if include_fda:
        fetch_fda_corpus(FDA_GUIDELINES, verbose=verbose)
    if verbose:
        print("Corpus generation complete. Check ./data/ directory.")

if __name__ == "__main__":
    # Example usage
    gather_biomed_corpus(
        pubmed_terms=["eligibility criteria oncology"], #PUBMED_DEFAULT_TERMS, (full) or "regulatory compliance AI" (specific)
        n_results=2,
        sleep=2.0,
        include_fda=False,
        verbose=True
    )