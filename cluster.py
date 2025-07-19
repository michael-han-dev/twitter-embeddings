import numpy as np
import json, os, umap, hdbscan
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

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
    unique_labels = np.array(labels)
    for label in unique_labels:
        if label == -1:
            continue
    cluster_docs = 

