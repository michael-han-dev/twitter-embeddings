import numpy as np
import json, os, umap, hdbscan
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as TFIDF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_selection import chi2

import chromadb

#initialize chromadb
client = chromadb.PersistentClient(path="./chroma.db")

def cluster(collection_name="tweets", username: str | None = None):
    collection = client.get_collection(collection_name)
    #filters username tweets to cluster
    where_clause = {"username": username} if username else {}

    result = collection.get(
        include=["embeddings", "documents"],
        where=where_clause
    )
    
    #convert list of list to array
    emb = np.array(result["embeddings"])
    docs = result["documents"]

    #reduce high dimension
    umap_embeddings = umap.UMAP(
        n_components=2,
        metric="cosine", #used cosine instead of euclidean for directionality and ignores magnitude of tweet
        min_dist=0.0,
        random_state=42
    ).fit_transform(emb)
    while True:
        val = input("\nMin cluster size (rec 10-15)?: ")
        if val.isdigit() and int(val) > 0:
            min_cluster_size = int(val)
            break
    #cluster the points
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean"
    ).fit_predict(umap_embeddings)
    return docs, umap_embeddings, labels

def analyze_cluster(docs, labels):
    #iterate over labels != -1 labelled from hdbscan
    unique_labels = np.unique(labels)
    for label in unique_labels:
        if label == -1:
            continue
        cluster_docs = [docs[i] for i in range(len(docs)) if labels[i] == label]

        vectorizer = CountVectorizer(
            ngram_range=(1,2),
            min_df=2,
            max_df=0.8,
            max_features=20000
        )
        transformed = vectorizer.fit_transform(cluster_docs)

        #cTD-IDF (cluster-level analysis)
        class_matrix = transformed.sum(axis=0).toarray()
        tfidf = TFIDF(norm=None).fit_transform(class_matrix)

        #chi score for significance compared to other clusters
        chi_scores, _ = chi2(transformed, [label] * len(cluster_docs))
        weighted_scores = tfidf.multiply(chi_scores / chi_scores.max())

        feature_names = vectorizer.get_feature_names_out()
        scores = weighted_scores.toarray().flatten()
        top_indices = np.argsort(scores)[::1][:5]
        top_keywords = [feature_names[i] for i in top_indices]
        

if __name__ == "__main__":
    docs, coords, labels = cluster()
    analyze_cluster(docs, labels)