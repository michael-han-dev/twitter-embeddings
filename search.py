import chromadb
import warnings
import os
from dotenv import load_dotenv
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction, OpenAIEmbeddingFunction

load_dotenv()
client = chromadb.PersistentClient(path="./chroma.db")

def collection_to_query():
    collections = client.list_collections()
    
    if len(collections) == 0:
        warnings.warn("No collections found. Run cluster.py first to create collections.")
        return None
    
    if len(collections) == 1:
        return collections[0].name
    
    print("\nAvailable collections:")
    for i, collection in enumerate(collections):
        print(f"{i}: {collection.name}")
    
    collection_index = int(input("Which collection to query (enter #): "))
    return collections[collection_index].name

def get_num_results():
    num_results = int(input("How many results to retrieve: "))
    while num_results < 1 or num_results > 20:
        num_results = int(input("Enter number between 1-20: "))
    return num_results

def get_user_filter():
    filter_choice = input("\nSearch:\n1. All tweets\n2. Specific user only\nChoice: ")
    while filter_choice not in ["1", "2"]:
        filter_choice = input("Enter 1 or 2: ")
    
    if filter_choice == "1":
        return None
    
    username = input("Enter username to filter by: ")
    return {"username": username}

def format_results(results):
    if len(results['ids'][0]) == 0:
        print("\nNo results found.\n")
        return
    
    print(f"\nFound {len(results['ids'][0])} results:\n")
    
    for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
        print(f"{i+1}. @{metadata['username']} ({metadata['created_at'][:10]})")
        print(f"   {doc}")
        print("-----------------")

def get_embedding_function(collection_name):
    if "openai" in collection_name.lower():
        openai_key = os.getenv("OPENAI_API_KEY")
        if not openai_key:
            raise ValueError("OPENAI_API_KEY not found in env")
        return OpenAIEmbeddingFunction(
            api_key=openai_key,
            model_name="text-embedding-3-small"
        )
    else:
        return SentenceTransformerEmbeddingFunction(
            model_name="all-mpnet-base-v2"
        )

def main():
    collection_name = collection_to_query()
    if not collection_name:
        return
    
    # Get the correct embedding function for this collection
    embedding_fn = get_embedding_function(collection_name)
    collection = client.get_collection(collection_name, embedding_function=embedding_fn)
    
    num_results = get_num_results()
    user_filter = get_user_filter()
    
    print(f"\nQuerying collection: {collection_name}")
    print("Type 'exit' to quit\n")
    
    while True:
        query = input("Enter search query: ")
        if query.lower() == "exit":
            break
        
        try:
            if user_filter:
                results = collection.query(
                    query_texts=[query],
                    n_results=num_results,
                    where=user_filter
                )
            else:
                results = collection.query(
                    query_texts=[query],
                    n_results=num_results
                )
            
            format_results(results)
            
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()