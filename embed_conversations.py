import chromadb
from collections import defaultdict
from extract_tweets import load_tweets_from_file
from datetime import datetime

def group_tweets_by_conversation(tweets):
    conversations = defaultdict(list)
    
    for tweet in tweets:
        conv_id = tweet['conversation_id']
        conversations[conv_id].append({
            'id': tweet['id'],
            'text': tweet['text'],
            'created_at': tweet['created_at'],
            'is_reply': tweet['is_reply'],
            'is_retweet': tweet['is_retweet']
        })
    
    for conv_id in conversations:
        conversations[conv_id].sort(key=lambda x: x['created_at'])
    
    return conversations

def format_conversation(tweets):
    if not tweets:
        return ""
    
    start_date = tweets[0]['created_at'][:10]
    end_date = tweets[-1]['created_at'][:10]
    
    if start_date == end_date:
        header = f"Conversation Date: {start_date}\n"
    else:
        header = f"Conversation Date: {start_date} - {end_date}\n"
    
    header += f"Conversation ID: {tweets[0]['id']}\n"
    header += "—" * 40 + "\n"
    
    conversation_text = header
    
    for tweet in tweets:
        time_str = tweet['created_at'][11:19]
        tweet_type = ""
        
        if tweet['is_retweet']:
            tweet_type = " (Retweet)"
        elif tweet['is_reply']:
            tweet_type = " (Reply)"
        
        conversation_text += f"Time: {time_str}{tweet_type}\n"
        conversation_text += f"Tweet: {tweet['text']}\n\n"
    
    return conversation_text

def embed_conversations(tweets=None, collection_name="conversations"):
    if tweets is None:
        tweets = load_tweets_from_file()
    
    if not tweets:
        print("No tweets found to process")
        return None, None
    
    client = chromadb.PersistentClient(path="./chroma_db")
    
    try:
        collection = client.get_collection(name=collection_name)
        print(f"Using existing collection: {collection_name}")
    except ValueError:
        collection = client.create_collection(name=collection_name)
        print(f"Created new collection: {collection_name}")
    
    conversations = group_tweets_by_conversation(tweets)
    
    documents = []
    metadatas = []
    ids = []
    
    for conv_id, conv_tweets in conversations.items():
        if len(conv_tweets) > 1:
            formatted_conv = format_conversation(conv_tweets)
            
            documents.append(formatted_conv)
            metadatas.append({
                "conversation_id": str(conv_id),
                "tweet_count": len(conv_tweets),
                "start_date": conv_tweets[0]['created_at'],
                "end_date": conv_tweets[-1]['created_at'],
                "has_replies": any(t['is_reply'] for t in conv_tweets),
                "has_retweets": any(t['is_retweet'] for t in conv_tweets)
            })
            ids.append(f"conv_{conv_id}")
    
    if documents:
        collection.upsert(
            documents=documents,
            metadatas=metadatas,
            ids=ids
        )
        print(f"Embedded {len(documents)} conversations into ChromaDB")
        client.persist()
    else:
        print("No multi-tweet conversations found")
    
    return client, collection

if __name__ == "__main__":
    client, collection = embed_conversations()
    if collection:
        count = collection.count()
        print(f"Collection contains {count} conversations")