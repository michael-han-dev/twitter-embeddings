import numpy as np
import json, os, umap, hdbscan
from dotenv import load_dotenv

load_dotenv()

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animate

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer as TFIDF
from sklearn.feature_selection import chi2

from embedTweets import (initialize_chroma, process_tweets_for_embedding, embed_tweets)
from animate import animate_tweets


import chromadb

#initialize chromadb
client = chromadb.PersistentClient(path="./chroma.db")

def cluster_single_user(collection_name="tweets", username: str | None = None):
    collection = client.get_collection(collection_name)

    result = collection.get(
        include=["embeddings", "documents", "metadatas"]
    )
    
    #convert list of list to array 
    emb = np.array(result["embeddings"])
    docs = result["documents"]
    metadata = result["metadatas"]
    
    if username and metadata and docs:
        filtered_indices = [i for i, m in enumerate(metadata) if m and m.get("username") == username]
        emb = emb[filtered_indices]
        docs = [docs[i] for i in filtered_indices]
        metadata = [metadata[i] for i in filtered_indices]

    #reduce high dimension
    umap_embeddings = umap.UMAP(
        n_components=3,
        metric="cosine", #used cosine instead of euclidean for directionality and ignores magnitude of tweet
        min_dist=0.0,
        random_state=42
    ).fit_transform(emb)
    
    while True:
        val = input(f"\nMin cluster size for {username} (rec 8-12)?: ")
        if val.isdigit() and int(val) > 0:
            min_cluster_size = int(val)
            break
    #cluster the points
    labels = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        metric="euclidean"
    ).fit_predict(umap_embeddings)
    return docs, umap_embeddings, labels, emb, metadata

def cluster_multiple_users(usernames: list[str], collection_name="tweets"):
    collection = client.get_collection(collection_name)
    
    result = collection.get(include=["embeddings", "documents", "metadatas"])
    all_emb = np.array(result["embeddings"])
    all_docs = result["documents"]
    all_metadata = result["metadatas"]
    
    user_indices = {}
    user_docs = {}
    user_metadata = {}
    user_embeddings = {}
    
    for user in usernames:
        if all_metadata and all_docs:
            indices = [i for i, m in enumerate(all_metadata) if m and m.get("username") == user]
            user_indices[user] = indices
            user_docs[user] = [all_docs[i] for i in indices]
            user_metadata[user] = [all_metadata[i] for i in indices]
            user_embeddings[user] = np.array(all_emb[indices])
    
    #combine all user embeddings
    combined_embeddings = []
    combined_docs = []
    combined_metadata = []
    user_slice_map = {}
    
    current_idx = 0
    for user in usernames:
        start_idx = current_idx
        user_emb_array = user_embeddings[user]
        end_idx = start_idx + user_emb_array.shape[0]
        user_slice_map[user] = (start_idx, end_idx)
        
        combined_embeddings.append(user_emb_array)
        combined_docs.extend(user_docs[user])
        combined_metadata.extend(user_metadata[user])
        current_idx = end_idx
    
    #run umap on all embeds
    combined_emb = np.vstack(combined_embeddings)
    shared_coords = np.array(umap.UMAP(
        n_components=3,
        metric="cosine",
        min_dist=0.0,
        random_state=42
    ).fit_transform(combined_emb))
    
    all_labels = np.full(shared_coords.shape[0], -1) 
    cluster_offset = 0
    user_cluster_info = {}
    
    while True:
        val = input(f"\nMin cluster size (rec 8-15)?: ")
        if val.isdigit() and int(val) > 0:
            min_cluster_size = int(val)
            break

    for user in usernames:
        start_idx, end_idx = user_slice_map[user]
        user_coords = shared_coords[start_idx:end_idx]
        
        #cluster user's points in shared space
        user_labels = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="euclidean"
        ).fit_predict(user_coords)
        
        adjusted_labels = user_labels.copy()
        adjusted_labels[user_labels >= 0] += cluster_offset
        all_labels[start_idx:end_idx] = adjusted_labels
        
        max_cluster = user_labels.max() if user_labels.max() >= 0 else -1
        user_cluster_info[user] = {
            'offset': cluster_offset,
            'max_cluster': max_cluster,
            'cluster_count': max_cluster + 1 if max_cluster >= 0 else 0,
            'slice': (start_idx, end_idx)
        }
        
        if max_cluster >= 0:
            cluster_offset += max_cluster + 1
    
    return combined_docs, shared_coords, all_labels, combined_emb, combined_metadata, user_cluster_info

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
        tfidf = TFIDF().fit_transform(class_matrix)

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

def plot_clusters_2d(labels, umap_embeddings, representatives, keywords, metadata):
    clustered = labels >= 0
    fig, ax = plt.subplots()
    usernames = [m["username"] for m in metadata]
    unique_users = list(set(usernames))
    user_colours = {unique_users[i]: ["black", "blue", "red", "green", "purple"][i % 5] for i in range(len(unique_users))}

    legend_handles = []
    
    for user in unique_users:
        user_mask = np.array([u == user for u in usernames])

        #plot the noise in grey
        if np.any(user_mask & ~clustered):
            ax.scatter(
                umap_embeddings[user_mask & ~clustered, 0],
                umap_embeddings[user_mask & ~clustered, 1],
                s=10, alpha=0.5, color="gray"
            )
        
        if np.any(user_mask & clustered):
            ax.scatter(
                umap_embeddings[user_mask & clustered, 0],
                umap_embeddings[user_mask & clustered, 1],
                c=labels[user_mask & clustered],
                s=30,
                cmap="tab10",
                edgecolors=user_colours[user],
                linewidth=1.0
            )
            
        #add user to legend with their edge color
        legend_handles.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='lightgray', markeredgecolor=user_colours[user],
                                    markersize=8, markeredgewidth=2, label=f"@{user}"))

    if np.any(~clustered):
        legend_handles.append(Line2D([0], [0], marker='o', color='gray', 
                                    markersize=6, alpha=0.5, label="Misc Tweets"))

    annotations = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        
        cluster_user = None
        for user, info in user_cluster_info.items():
            if info['offset'] <= lab <= info['offset'] + info['max_cluster']:
                cluster_user = user
                break
        
        pts = umap_embeddings[labels == lab]
        centroid = pts.mean(0)
        idx = np.where(labels == lab)[0][np.argmin(np.linalg.norm(pts - centroid, axis=1))]
        x, y = umap_embeddings[idx][:2]
        
        box_edge_color = user_colours.get(cluster_user, "black") if cluster_user else "black"
        
        ann = ax.annotate(
            representatives[lab][:70] + "…",
            (x, y),
            textcoords="offset points",
            xytext=(0, 3),
            ha="center",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor=box_edge_color, linewidth=1.5),
            color="black"
        )
        annotations[lab] = ann

    ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize="small")

    state = {"rep": True, "show_all": False}
    tweet_annotations = []

    def on_key(event):
        if event.key == "t":
            state["rep"] = not state["rep"]
            for lab, ann in annotations.items():
                ann.set_text(
                    keywords[lab] if not state["rep"] else representatives[lab][:70] + "…"
                )
        elif event.key == "r":
            state["show_all"] = not state["show_all"]
            
            for ann in tweet_annotations:
                ann.remove()
            tweet_annotations.clear()
            
            if state["show_all"]:
                for i, (coord, meta) in enumerate(zip(umap_embeddings, metadata)):
                    if meta and 'text' in meta:
                        user = meta['username']
                        edge_color = user_colours.get(user, "black")
                        tweet_text = f"@{user}: {meta['text'][:50]}..."
                        
                        ann = ax.annotate(tweet_text, (coord[0], coord[1]),
                                         xytext=(5, 5), textcoords='offset points',
                                         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=edge_color),
                                         fontsize=5, ha='left')
                        tweet_annotations.append(ann)
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_title("Tweet clusters (UMAP 2-D)")
    ax.set_xlabel("'t' to switch cluster rep/keyword, 'r' to toggle all tweets")
    ax.set_ylabel("UMAP-2")
    plt.tight_layout()
    plt.show()

def plot_clusters_3d(labels, umap_embeddings, representatives, keywords, metadata):
    clustered = labels >= 0
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(False)
    ax.set_facecolor('none')
    ax.set_axis_off()

    usernames = [m["username"] for m in metadata]
    unique_users = list(set(usernames))
    user_colours = {unique_users[i]: ["black", "blue", "red", "green", "purple"][i % 5] for i in range(len(unique_users))}

    legend_handles = []
    
    for user in unique_users:
        user_mask = np.array([u == user for u in usernames])

        #plot the noise in grey
        if np.any(user_mask & ~clustered):
            ax.scatter(
                umap_embeddings[user_mask & ~clustered, 0],
                umap_embeddings[user_mask & ~clustered, 1],
                umap_embeddings[user_mask & ~clustered, 2],
                alpha=0.5, color="gray"
            )
        
        if np.any(user_mask & clustered):
            ax.scatter(
                umap_embeddings[user_mask & clustered, 0],
                umap_embeddings[user_mask & clustered, 1],
                umap_embeddings[user_mask & clustered, 2],
                c=labels[user_mask & clustered],
                s=30,
                cmap="tab10",
                edgecolors=user_colours[user],
                linewidth=1.5
            )
        
        legend_handles.append(Line2D([0], [0], marker='o', color='w', 
                                    markerfacecolor='lightgray', markeredgecolor=user_colours[user],
                                    markersize=8, markeredgewidth=2, label=f"@{user}"))
    
    if np.any(~clustered):
        legend_handles.append(Line2D([0], [0], marker='o', color='gray', 
                                    markersize=6, alpha=0.5, label="Misc Tweets"))

    annotations = {}
    for lab in np.unique(labels):
        if lab == -1:
            continue
        
        cluster_user = None
        for user, info in user_cluster_info.items():
            if info['offset'] <= lab <= info['offset'] + info['max_cluster']:
                cluster_user = user
                break
        
        pts = umap_embeddings[labels == lab]
        centroid = pts.mean(0)
        idx = np.where(labels == lab)[0][np.argmin(np.linalg.norm(pts - centroid, axis=1))]
        x, y, z = umap_embeddings[idx]
        
        box_edge_color = user_colours.get(cluster_user, "black") if cluster_user else "black"
        
        ann = ax.text(
            x, y, z,
            representatives[lab][:70] + "…",
            ha="center",
            va="bottom",
            fontsize=6,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7, edgecolor=box_edge_color, linewidth=1.5),
            color="black"
        )
        annotations[lab] = ann

    ax.legend(handles=legend_handles, loc="best", frameon=False, fontsize="small")

    state = {"rep": True, "show_all": False}
    tweet_annotations = []

    def on_key(event):
        if event.key == "t":
            state["rep"] = not state["rep"]
            for lab, ann in annotations.items():
                ann.set_text(
                    keywords[lab] if not state["rep"] else representatives[lab][:70] + "…"
                )
        elif event.key == "r":
            state["show_all"] = not state["show_all"]
            
            for ann in tweet_annotations:
                ann.remove()
            tweet_annotations.clear()
            
            if state["show_all"]:
                for coord, meta in zip(umap_embeddings, metadata):
                    if meta and 'text' in meta:
                        user = meta['username']
                        edge_color = user_colours.get(user, "black")
                        tweet_text = f"@{user}: {meta['text'][:50]}..."
                        
                        ann = ax.text(coord[0], coord[1], coord[2], tweet_text,
                                     bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor=edge_color),
                                     fontsize=5, ha='center')
                        tweet_annotations.append(ann)
        
        fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    ax.set_title("Tweet clusters (UMAP 3-D)")
    ax.set_xlabel("'t' to switch cluster rep/keyword, 'r' to toggle all tweets")
    ax.set_ylabel("UMAP-2")
    ax.set_zlabel("UMAP-3")
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

    # Embed tweets
    print("Embedding tweets...")
    collection_name = f"tweets_{provider}"
    client, collection = initialize_chroma(collection_name, provider=provider, openai_key=openai_key)
    docs, metadata, ids = process_tweets_for_embedding("tweets.json")
    embed_tweets(collection, docs, metadata, ids)
    
    print("Would you like to analyze:")
    print("1. One user")  
    print("2. Two users")

    while True:
        choice = input("Enter choice (1 or 2): ").strip()
        if choice == "1":
            while True:
                username = input("Enter username to be clustered: ")
                if username and username in valid_usernames:
                    usernames = [username]
                    break
                print("Not a valid username or user tweets non-existent.")
            break
        elif choice == "2":
            usernames = []
            for i in range(2):
                while True:
                    username = input(f"Enter username {i+1} to be clustered: ")
                    if username and username in valid_usernames:
                        usernames.append(username)
                        break
                    print("Not a valid username or user tweets non-existent.")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")
        
    # Cluster and analyze
    print("Clustering...")
    docs, coords, labels, emb, metadata, user_cluster_info = cluster_multiple_users(usernames, collection_name)
    keywords = analyze_cluster(docs, labels)
    rep = cluster_representation(docs, emb, labels)
    
    print("Visualize in:")
    print("1. 2D Static")
    print("2. 3D Static")
    print("3. 2D Animated Timeline")
    print("4. 3D Animated Timeline")
    
    while True:
        viz_choice = input("Enter choice (1-4): ").strip()
        if viz_choice == "1":
            plot_clusters_2d(labels, coords, rep, keywords, metadata)
            break
        elif viz_choice == "2":
            plot_clusters_3d(labels, coords, rep, keywords, metadata)
            break
        elif viz_choice == "3":
            animate_tweets(labels, coords, metadata, mode='2d')
            break
        elif viz_choice == "4":
            animate_tweets(labels, coords, metadata, mode='3d')
            break
        else:
            print("Invalid choice. Please enter 1-4.")