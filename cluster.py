import numpy as np
import json, os, umap, hdbscan
from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as TFIDF
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
    return docs, umap_embeddings, labels, emb
def analyze_cluster(docs, labels):
    #iterate over labels != -1 labelled from hdbscan
    keywords = {}
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

        scores = weighted_scores.toarray().flatten()
        top_indices = np.argsort(scores)[::1][:2]
        keywords[label] = ", ".join(vectorizer.get_feature_names_out()[top_indices])
    return keywords

def cluster_representation(docs, embeddings, labels):
    #get the clusters and calculate centroid tweet, use that as label for the cluster.
    unique_labels = np.unique(labels)
    representatives = {}
    for label in unique_labels:
        if label == -1:
            continue
        index = np.where(labels==label)[0]
        cluster_embeddings = embeddings[index]
        centroid = np.mean(cluster_embeddings, axis=0) 
        distance = np.linalg.norm(cluster_embeddings - centroid, axis=1)
        closet = index[np.argmin(distance)]
        representatives[label] = docs[closet]
        print(f"Cluster {label} ({len(index)} tweets) representative: {docs[closet]}")
    
    return representatives

def plot_clusters(labels, umap_embeddings, representatives, keywords):
    clustered = labels >= 0
    fig, ax = plt.subplots()

    #plot the noise in greey
    noise = ax.scatter(
        umap_embeddings[~clustered, 0],
        umap_embeddings[~clustered, 1],
        s=10, alpha=0.5, color="gray", label="Misc Tweets"
    )
    ax.scatter(
        umap_embeddings[clustered, 0],
        umap_embeddings[clustered, 1],
        c=labels[clustered], s=10, cmap="Spectral"
    )

    annotations = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        pts = umap_embeddings[labels == lab]
        centroid = pts.mean(0)
        idx = np.where(labels == lab)[0][np.argmin(np.linalg.norm(pts - centroid, axis=1))]
        x, y = umap_embeddings[idx]
        ann = ax.annotate(
            representatives[lab][:70] + "…",
            (x, y),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7),
            color="black"
        )
        annotations[lab] = ann

    ax.legend(handles=[noise], loc="best", frameon=False, fontsize="small")

    state = {"rep": True}

    def on_key(event):
        if event.key != "t":
            return
        state["rep"] = not state["rep"]
        for lab, ann in annotations.items():
            ann.set_text(
                keywords[lab] if not state["rep"] else representatives[lab][:70] + "…"
            )
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_title("Tweet clusters (UMAP 2-D)")
    ax.set_xlabel("'t' to switch between cluster rep tweet or keyword")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

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
            openai_key = os.getenv("OPENAI_API_KEY")
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
    docs, coords, labels, emb = cluster(username=username)
    keywords = analyze_cluster(docs, labels)
    rep = cluster_representation(docs, emb, labels)
    plot_clusters(labels, coords, rep, keywords)