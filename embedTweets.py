import chromadb
from chromadb.utils import embedding_functions
import json
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from extractTW import load_tweets

def initialize_chroma(collection_name="tweets"):
    client = chromadb.PersistentClient(path="./chroma.db")
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using an existing collection {collection_name}")
    except:
        collection = client.create_collection(name=collection_name)
        print(f"Using new collection {collection_name}")
    
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
        metadata.append({"created_at":t["created_at"], "username":t["username"]})
        ids.append(str(t["id"]))
    return documents, metadata, ids

#embed the tweets using upsert in case new embedding function is used
def embed_tweets(collection, documents, metadata, ids, provider="local", openai_key=None):
    #set flag for using openai embedding or default embedding from sentence_transformer
    if provider == "openai":
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
        embeddings = embedding_fn(documents)
    else:
        embedding_fn = SentenceTransformer("all-mpnet-base-v2")
        embeddings = embedding_fn.encode(documents, show_progress_bar=True)
    collection.upsert(documents=documents, embeddings=embeddings, metadatas=metadata, ids=ids)


