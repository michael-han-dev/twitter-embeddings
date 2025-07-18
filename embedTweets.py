import chromadb
from chromadb.utils import embedding_functions
import json

from llvmlite.ir import Value
from extractTW import load_tweets

def initialize_chroma(collection_name="tweets"):
    client = chromadb.PersistentClient(path="./chroma.db")
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using an existing collection {collection_name}")
    except ValueError:
        collection = client.get_collection(name=collection_name)
        print(f"Using new collection {collection_name}")
    
    return client, collection

def process_tweets_for_embedding(tweets_json: str):
    