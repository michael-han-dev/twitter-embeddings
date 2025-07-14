#!/usr/bin/env python3

import os
import sys
from extract_tweets import get_user_tweets, save_tweets_to_file, load_tweets_from_file
from embed_tweets import embed_tweets, get_collection_stats
from embed_conversations import embed_conversations
from search import interactive_search, search_tweets, format_search_results
from cluster import run_clustering_analysis

def print_menu():
    print("\n" + "="*50)
    print("TWITTER EMBEDDINGS APP")
    print("="*50)
    print("1. Extract tweets from Twitter API")
    print("2. Embed tweets into vector database")
    print("3. Embed conversations")
    print("4. Search tweets")
    print("5. Cluster analysis")
    print("6. Quick search (non-interactive)")
    print("7. Show collection stats")
    print("0. Exit")
    print("-"*50)

def extract_tweets_menu():
    username = input("Enter Twitter username (without @): ").strip()
    if not username:
        print("Username required")
        return
    
    try:
        max_tweets = int(input("Max tweets to extract (default 500): ") or "500")
    except ValueError:
        max_tweets = 500
    
    try:
        print(f"Extracting tweets for @{username}...")
        tweets = get_user_tweets(username, max_tweets)
        save_tweets_to_file(tweets)
        print(f"Successfully extracted {len(tweets)} tweets")
    except Exception as e:
        print(f"Error extracting tweets: {e}")

def embed_tweets_menu():
    tweets = load_tweets_from_file()
    if not tweets:
        print("No tweets found. Extract tweets first.")
        return
    
    print(f"Embedding {len(tweets)} tweets...")
    client, collection = embed_tweets(tweets)
    if collection:
        get_collection_stats(collection)

def embed_conversations_menu():
    tweets = load_tweets_from_file()
    if not tweets:
        print("No tweets found. Extract tweets first.")
        return
    
    print("Creating conversation embeddings...")
    client, collection = embed_conversations(tweets)

def search_menu():
    print("Starting interactive search...")
    interactive_search()

def quick_search_menu():
    query = input("Enter search query: ").strip()
    if not query:
        return
    
    try:
        n_results = int(input("Number of results (default 5): ") or "5")
    except ValueError:
        n_results = 5
    
    results = search_tweets(query, n_results)
    format_search_results(results)

def cluster_menu():
    try:
        min_size = int(input("Minimum cluster size (default 5): ") or "5")
    except ValueError:
        min_size = 5
    
    run_clustering_analysis(min_size)

def show_stats_menu():
    from embed_tweets import initialize_chroma
    
    try:
        client, collection = initialize_chroma("tweets")
        print("\nTweets Collection:")
        get_collection_stats(collection)
    except:
        print("No tweets collection found")
    
    try:
        client, collection = initialize_chroma("conversations")
        count = collection.count()
        print(f"\nConversations Collection: {count} conversations")
    except:
        print("No conversations collection found")

def setup_environment():
    if not os.path.exists('.env'):
        print("Setting up environment...")
        bearer_token = input("Enter your Twitter Bearer Token: ").strip()
        
        with open('.env', 'w') as f:
            f.write(f"TWITTER_BEARER_TOKEN={bearer_token}\n")
        
        print("Environment setup complete!")
        print("\nTo get a Twitter Bearer Token:")
        print("1. Go to https://developer.twitter.com/")
        print("2. Create a developer account and app")
        print("3. Generate Bearer Token in your app settings")

def main():
    if not os.path.exists('.env'):
        setup_environment()
    
    while True:
        try:
            print_menu()
            choice = input("Select option: ").strip()
            
            if choice == '0':
                print("Goodbye!")
                break
            elif choice == '1':
                extract_tweets_menu()
            elif choice == '2':
                embed_tweets_menu()
            elif choice == '3':
                embed_conversations_menu()
            elif choice == '4':
                search_menu()
            elif choice == '5':
                cluster_menu()
            elif choice == '6':
                quick_search_menu()
            elif choice == '7':
                show_stats_menu()
            else:
                print("Invalid option")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()