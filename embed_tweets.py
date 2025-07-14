import chromadb
from chromadb.utils import embedding_functions
import json
from extract_tweets import load_tweets_from_file
from datetime import datetime

def initialize_chroma(collection_name="tweets"):
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except ValueError:
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
    
    return client, collection

def process_tweets_for_embedding(tweets):
    documents = []
    metadatas = []
    ids = []
    
    for tweet in tweets:
        if tweet['text'] and tweet['text'].strip():
            text = tweet['text']
            if tweet['is_retweet']:
                text = text.replace('RT @', 'Retweeted from @')
            
            documents.append(text)
            metadatas.append({
                "tweet_id": str(tweet['id']),
                "created_at": tweet['created_at'],
                "conversation_id": str(tweet['conversation_id']),
                "is_reply": tweet['is_reply'],
                "is_retweet": tweet['is_retweet'],
                "retweet_count": tweet['public_metrics']['retweet_count'],
                "like_count": tweet['public_metrics']['like_count']
            })
            ids.append(str(tweet['id']))
    
    return documents, metadatas, ids

def embed_tweets(tweets=None, collection_name="tweets"):
    if tweets is None:
        tweets = load_tweets_from_file()
    
    if not tweets:
        print("No tweets found to embed")
        return None, None
    
    client, collection = initialize_chroma(collection_name)
    
    documents, metadatas, ids = process_tweets_for_embedding(tweets)
    
    if documents:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Embedded {len(documents)} tweets into ChromaDB")
        client.persist()
    
    return client, collection

def get_collection_stats(collection):
    count = collection.count()
    sample = collection.peek(limit=3)
    
    print(f"Collection contains {count} tweets")
    if sample['documents']:
        print("\nSample tweets:")
        for i, (doc, meta) in enumerate(zip(sample['documents'], sample['metadatas'])):
            print(f"{i+1}. {doc[:100]}...")
            print(f"   Date: {meta['created_at']}, Likes: {meta['like_count']}")

if __name__ == "__main__":
    client, collection = embed_tweets()
    if collection:
        get_collection_stats(collection)