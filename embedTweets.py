import chromadb
import json
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction

from extractTW import load_tweets

def initialize_chroma(collection_name="tweets", provider="local", openai_key=None):
    client = chromadb.PersistentClient(path="./chroma.db")
    
    if provider == "openai":
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
    else:
        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )
    
    collection = client.get_or_create_collection(
        name=collection_name,
        embedding_function=embedding_fn
    )
    print(f"Using collection {collection_name} with {provider} embeddings")
    
    return client, collection

def process_tweets_for_embedding(tweets_json: str):
    tweets = load_tweets(tweets_json)
    
    documents = []
    metadata = []
    ids = []

    #load tweets into respective lists
    for t in tweets:
        cleaned = t["text"].replace("\n", " ").strip()
        if not cleaned:
            continue
        documents.append(cleaned)
        metadata.append({"created_at":t["created_at"], "username":t["username"], "text":cleaned})
        ids.append(str(t["id"]))
    return documents, metadata, ids

#embed the tweets using upsert in case new embedding function is used
def embed_tweets(collection, documents, metadata, ids):
    collection.upsert(documents=documents, metadatas=metadata, ids=ids)


