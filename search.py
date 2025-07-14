import chromadb
from datetime import datetime

def get_collection(collection_name=None):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    collections = client.list_collections()
    if not collections:
        print("No collections found. Run embed_tweets.py first.")
        return None, None
    
    if collection_name:
        try:
            collection = client.get_collection(name=collection_name)
            return client, collection
        except ValueError:
            print(f"Collection '{collection_name}' not found")
            return None, None
    
    if len(collections) == 1:
        collection_name = collections[0].name
        collection = client.get_collection(name=collection_name)
        print(f"Using collection: {collection_name}")
        return client, collection
    
    print("Available collections:")
    for i, coll in enumerate(collections):
        print(f"{i+1}. {coll.name}")
    
    while True:
        try:
            choice = int(input("Select collection (number): ")) - 1
            if 0 <= choice < len(collections):
                collection_name = collections[choice].name
                collection = client.get_collection(name=collection_name)
                return client, collection
            else:
                print("Invalid choice")
        except (ValueError, KeyboardInterrupt):
            return None, None

def search_tweets(query, n_results=10, collection_name=None, date_filter=None):
    client, collection = get_collection(collection_name)
    if not collection:
        return []
    
    where_clause = {}
    if date_filter:
        if isinstance(date_filter, dict):
            where_clause.update(date_filter)
    
    try:
        if where_clause:
            results = collection.query(
                query_texts=[query],
                n_results=n_results,
                where=where_clause
            )
        else:
            results = collection.query(
                query_texts=[query],
                n_results=n_results
            )
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def format_search_results(results, show_metadata=True):
    if not results or not results['documents']:
        print("No results found")
        return
    
    documents = results['documents'][0]
    metadatas = results['metadatas'][0] if results['metadatas'] else [{}] * len(documents)
    distances = results['distances'][0] if results['distances'] else [0] * len(documents)
    
    print(f"\nFound {len(documents)} results:\n")
    
    for i, (doc, meta, dist) in enumerate(zip(documents, metadatas, distances)):
        print(f"Result {i+1} (similarity: {1-dist:.3f}):")
        print(f"Text: {doc[:200]}{'...' if len(doc) > 200 else ''}")
        
        if show_metadata and meta:
            if 'created_at' in meta:
                date_str = meta['created_at'][:10] if meta['created_at'] else 'Unknown'
                print(f"Date: {date_str}")
            
            if 'like_count' in meta:
                print(f"Likes: {meta['like_count']}, Retweets: {meta.get('retweet_count', 0)}")
            
            if 'tweet_count' in meta:
                print(f"Conversation with {meta['tweet_count']} tweets")
        
        print("-" * 50)

def interactive_search():
    print("Twitter Embedding Search")
    print("Type 'quit' to exit, 'collections' to switch collections")
    
    client, collection = get_collection()
    if not collection:
        return
    
    while True:
        try:
            query = input("\nEnter search query: ").strip()
            
            if query.lower() == 'quit':
                break
            elif query.lower() == 'collections':
                client, collection = get_collection()
                if not collection:
                    break
                continue
            elif not query:
                continue
            
            try:
                n_results = int(input("Number of results (default 5): ") or "5")
            except ValueError:
                n_results = 5
            
            results = search_tweets(query, n_results, collection.name)
            format_search_results(results)
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break

if __name__ == "__main__":
    interactive_search()