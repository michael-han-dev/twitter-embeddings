import numpy as np
import json, os, umap, hdbscan
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as TFIDF
from sklearn.decomposition import LatentDirichletAllocation as LDA
from sklearn.feature_selection import chi2
from embedTweets import (initialize_chroma, process_tweets_for_embedding, embed_tweets)


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
            max_features=20000,
            stop_words="english",
            token_pattern = r"(?u)\b[a-zA-Z]{2,}\b"
        )
        transformed = vectorizer.fit_transform(cluster_docs)

        #cTD-IDF (cluster-level analysis)
        class_matrix = np.asarray(transformed.sum(axis=0))
        tfidf = TFIDF(norm=None).fit_transform(class_matrix)

        #chi score for significance compared to other clusters
        chi_scores, _ = chi2(transformed, [label] * len(cluster_docs))
        weighted_scores = tfidf.multiply(chi_scores / chi_scores.max())

        feature_names = vectorizer.get_feature_names_out()
        scores = weighted_scores.toarray().flatten()
        top_indices = np.argsort(scores)[::1][:2]
        top_keywords = [feature_names[i] for i in top_indices]
        print(f"Cluster {label} ({len(cluster_docs)} docs): {top_keywords}")

def cluster_representation(docs, umap_embeddings, labels):
    #get the clusters and calculate centroid tweet, use that as label for the cluster.


if __name__ == "__main__":
    # Provider selection
    print("Select embedding provider:")
    print("1. Local model (sentence-transformers)")
    print("2. OpenAI API")
    
    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            provider = "local"
            openai_key = None
            break
        elif choice == "2":
            provider = "openai"
            openai_key = input("Enter OpenAI API key: ").strip()
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

    with open("tweets.json", "r") as f:
        tweets = json.load(f)
        valid_usernames = set(tweet["username"] for tweet in tweets)
    while True:
        username = input("Enter username to be clustered: ")
        if username and username in valid_usernames:
            break
        print("Not a valid username or user tweets non-existent.")

    # Embed tweets
    print("Embedding tweets...")
    client, collection = initialize_chroma("tweets")
    docs, metadata, ids = process_tweets_for_embedding("tweets.json")
    embed_tweets(collection, docs, metadata, ids, provider=provider, openai_key=openai_key)
    
    # Cluster and analyze
    print("Clustering...")
    docs, coords, labels = cluster(username=username)
    analyze_cluster(docs, labels)