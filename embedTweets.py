import chromadb
from chromadb.utils import embedding_functions
import json
from sentence_transformers import SentenceTransformer
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from llvmlite.ir import Value
from extractTW import load_tweets

def initialize_chroma(collection_name="tweets", provider="default", openai_key=None):
    client = chromadb.PersistentClient(path="./chroma.db")
    if provider == "openai":
        embedding_fn = OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
    else:
        embedding_fn = None
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using an existing collection {collection_name}")
    except ValueError:
        collection = client.create_collection(name=collection_name, 
        embedding_function=embedding_fn
        )
        print(f"Using new collection {collection_name}")
    
    return client, collection

def process_tweets_for_embedding(tweets_json: str):
    with open(tweets_json, "r") as f:
        tweets = json.load(f)
    
    documents = []
    metadata = []
    ids = []

    for t in tweets:
        cleaned = t["text"].replace("\n", " ").strip()
        if not cleaned:
            continue
        documents.append(cleaned)
        metadata.append({"created_at":t["created_at"]})
        ids.append(str(t["id"]))
    return documents, metadata, ids


