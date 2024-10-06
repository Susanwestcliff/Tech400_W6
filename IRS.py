import os
import re
import numpy as np
from collections import defaultdict
from math import log

# Preprocessing
def clean_text(text):
    return re.findall(r'\b\w+\b', text.lower())

# Load text files into a dictionary
def fetch_documents(directory_path):
    documents = {}
    for doc_file in os.listdir(directory_path):
        if doc_file.endswith('.txt'):
            with open(os.path.join(directory_path, doc_file), 'r', encoding='utf-8', errors='replace') as f:
                documents[doc_file] = clean_text(f.read())
    return documents

# Loading query
def fetch_queries(query_path):
    with open(query_path, 'r', encoding='utf-8', errors='replace') as f:
        return [line.strip() for line in f.readlines()]

# Calculate term frequencies and document frequencies
def calculate_frequencies(docs):
    total_docs = len(docs)
    term_in_docs = defaultdict(int)
    term_frequency = defaultdict(lambda: defaultdict(int))

    for document, terms in docs.items():
        unique_terms = set(terms)
        for term in terms:
            term_frequency[document][term] += 1
        for term in unique_terms:
            term_in_docs[term] += 1

    return term_frequency, term_in_docs, total_docs

# Compute BIM-based relevance scores
def get_relevance_score(query_terms, term_frequency, term_in_docs, total_docs):
    relevance_scores = {}
    for doc_name in term_frequency:
        probability_score = 0.5
        for term in query_terms:
            term_freq = term_frequency[doc_name].get(term, 0)
            doc_freq = term_in_docs.get(term, 0)
            prob_relevant = (term_freq + 1) / (sum(term_frequency[doc_name].values()) + len(term_in_docs))
            prob_non_relevant = (doc_freq + 1) / (total_docs - doc_freq + len(term_in_docs))
            probability_score *= (prob_relevant / prob_non_relevant)
        relevance_scores[doc_name] = probability_score
    return relevance_scores

# Core function to retrieve documents based on queries
def search_documents(directory_path, query_path):
    documents = fetch_documents(directory_path)
    query_list = fetch_queries(query_path)

    term_frequency, term_in_docs, total_docs = calculate_frequencies(documents)

    for query_text in query_list:
        query_words = clean_text(query_text)
        scores = get_relevance_score(query_words, term_frequency, term_in_docs, total_docs)
        sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        print(f"Query: {query_text}")
        for doc_name, rel_score in sorted_docs:
            print(f"Document: {doc_name}, Score: {rel_score:.4f}") 
        print()

# Directory setup (adjust these paths as needed)
doc_directory = './final_data'  
query_file = './final_query/final_query.txt'

# Call the search function
search_documents(doc_directory, query_file)