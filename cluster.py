import chromadb
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import umap
import hdbscan

def get_tweet_embeddings(collection_name="tweets", filter_retweets=True):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = client.get_collection(name=collection_name)
    except ValueError:
        print(f"Collection '{collection_name}' not found")
        return None, None, None
    
    where_clause = {}
    if filter_retweets:
        where_clause["is_retweet"] = False
    
    if where_clause:
        results = collection.get(
            include=['embeddings', 'documents', 'metadatas'],
            where=where_clause
        )
    else:
        results = collection.get(
            include=['embeddings', 'documents', 'metadatas']
        )
    
    if not results['embeddings']:
        print("No embeddings found")
        return None, None, None
    
    embeddings = np.array(results['embeddings'])
    documents = results['documents']
    metadatas = results['metadatas']
    
    return embeddings, documents, metadatas

def cluster_tweets(min_cluster_size=5, filter_retweets=True):
    embeddings, documents, metadatas = get_tweet_embeddings(filter_retweets=filter_retweets)
    
    if embeddings is None:
        return None, None, None, None
    
    print(f"Clustering {len(embeddings)} tweets...")
    
    umap_reducer = umap.UMAP(
        n_neighbors=min(30, len(embeddings)-1),
        min_dist=0.0,
        n_components=2,
        random_state=42
    )
    
    reduced_embeddings = umap_reducer.fit_transform(embeddings)
    
    clusterer = hdbscan.HDBSCAN(
        min_samples=10,
        min_cluster_size=min_cluster_size
    )
    
    cluster_labels = clusterer.fit_predict(reduced_embeddings)
    
    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = list(cluster_labels).count(-1)
    
    print(f"Found {n_clusters} clusters, {n_noise} outliers")
    
    return cluster_labels, reduced_embeddings, documents, metadatas

def analyze_clusters(cluster_labels, documents, metadatas, top_words=5):
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)
    
    cluster_analysis = {}
    
    for cluster_id in unique_labels:
        cluster_mask = np.array(cluster_labels) == cluster_id
        cluster_docs = [doc for i, doc in enumerate(documents) if cluster_mask[i]]
        cluster_metas = [meta for i, meta in enumerate(metadatas) if cluster_mask[i]]
        
        if not cluster_docs:
            continue
        
        vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        try:
            tfidf_matrix = vectorizer.fit_transform(cluster_docs).toarray()  # type: ignore[attr-defined]
            feature_names = vectorizer.get_feature_names_out()
            
            mean_tfidf = np.mean(tfidf_matrix, axis=0)
            top_indices = np.argsort(mean_tfidf)[::-1][:top_words]
            top_keywords = [feature_names[i] for i in top_indices]
            
            lda = LatentDirichletAllocation(n_components=1, random_state=42)
            lda.fit(tfidf_matrix)
            
            topic_words = []
            for topic_idx, topic in enumerate(lda.components_):
                top_word_indices = topic.argsort()[::-1][:top_words]
                topic_words.extend([feature_names[i] for i in top_word_indices])
            
        except ValueError:
            top_keywords = []
            topic_words = []
        
        avg_likes = np.mean([meta.get('like_count', 0) for meta in cluster_metas])
        avg_retweets = np.mean([meta.get('retweet_count', 0) for meta in cluster_metas])
        
        cluster_analysis[cluster_id] = {
            'size': len(cluster_docs),
            'keywords': top_keywords,
            'topics': topic_words[:top_words],
            'avg_likes': avg_likes,
            'avg_retweets': avg_retweets,
            'sample_tweets': cluster_docs[:3]
        }
    
    return cluster_analysis

def find_cluster_representatives(cluster_labels, reduced_embeddings, documents):
    unique_labels = set(cluster_labels)
    unique_labels.discard(-1)
    
    representatives = {}
    
    for cluster_id in unique_labels:
        cluster_mask = np.array(cluster_labels) == cluster_id
        cluster_points = reduced_embeddings[cluster_mask]
        cluster_docs = [doc for i, doc in enumerate(documents) if cluster_mask[i]]
        
        if len(cluster_points) == 0:
            continue
        
        centroid = np.mean(cluster_points, axis=0)
        distances = np.linalg.norm(cluster_points - centroid, axis=1)
        closest_idx = np.argmin(distances)
        
        representatives[cluster_id] = cluster_docs[closest_idx]
    
    return representatives

def visualize_clusters(cluster_labels, reduced_embeddings, save_path="clusters.png"):
    plt.figure(figsize=(12, 8))
    
    unique_labels = set(cluster_labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    for cluster_id, color in zip(unique_labels, colors):
        if cluster_id == -1:
            color = 'black'
            marker = 'x'
            label = 'Outliers'
        else:
            marker = 'o'
            label = f'Cluster {cluster_id}'
        
        cluster_mask = np.array(cluster_labels) == cluster_id
        plt.scatter(
            reduced_embeddings[cluster_mask, 0],
            reduced_embeddings[cluster_mask, 1],
            c=[color],
            marker=marker,
            label=label,
            alpha=0.7,
            s=50
        )
    
    plt.title('Tweet Clusters (UMAP + HDBSCAN)')
    plt.xlabel('UMAP Dimension 1')
    plt.ylabel('UMAP Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Cluster visualization saved to {save_path}")

def run_clustering_analysis(min_cluster_size=5):
    cluster_labels, reduced_embeddings, documents, metadatas = cluster_tweets(min_cluster_size)
    
    if cluster_labels is None:
        return
    
    print("\n" + "="*50)
    print("CLUSTER ANALYSIS")
    print("="*50)
    
    analysis = analyze_clusters(cluster_labels, documents, metadatas)
    representatives = find_cluster_representatives(cluster_labels, reduced_embeddings, documents)
    
    for cluster_id in sorted(analysis.keys()):
        cluster_info = analysis[cluster_id]
        print(f"\nCluster {cluster_id} ({cluster_info['size']} tweets):")
        print(f"Keywords: {', '.join(cluster_info['keywords'])}")
        print(f"Topics: {', '.join(cluster_info['topics'])}")
        print(f"Avg Engagement: {cluster_info['avg_likes']:.1f} likes, {cluster_info['avg_retweets']:.1f} retweets")
        
        if cluster_id in representatives:
            print(f"Representative: {representatives[cluster_id][:100]}...")
        
        print("\nSample tweets:")
        for i, tweet in enumerate(cluster_info['sample_tweets']):
            print(f"  {i+1}. {tweet[:80]}...")
    
    visualize_clusters(cluster_labels, reduced_embeddings)

if __name__ == "__main__":
    print("Starting clustering analysis...")
    try:
        min_size = int(input("Minimum cluster size (default 5): ") or "5")
    except ValueError:
        min_size = 5
    
    run_clustering_analysis(min_size)