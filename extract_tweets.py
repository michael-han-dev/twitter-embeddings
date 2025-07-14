import tweepy
import os
from dotenv import load_dotenv
from datetime import datetime
import json

load_dotenv()

def get_twitter_client():
    bearer_token = os.getenv('TWITTER_BEARER_TOKEN')
    if not bearer_token:
        raise ValueError("TWITTER_BEARER_TOKEN not found in environment variables")
    
    return tweepy.Client(bearer_token=bearer_token)

def get_user_tweets(username, max_results=100):
    client = get_twitter_client()
    
    user = client.get_user(username=username)
    if not user.data:
        raise ValueError(f"User {username} not found")
    
    user_id = user.data.id
    
    tweets = []
    try:
        paginator = tweepy.Paginator(
            client.get_users_tweets,
            id=user_id,
            tweet_fields=['created_at', 'public_metrics', 'context_annotations', 'conversation_id', 'in_reply_to_user_id', 'referenced_tweets'],
            max_results=100,
            limit=max_results//100 + 1
        )
        
        for page in paginator:
            if page.data:
                for tweet in page.data:
                    tweet_data = {
                        'id': tweet.id,
                        'text': tweet.text,
                        'created_at': tweet.created_at.isoformat(),
                        'conversation_id': tweet.conversation_id,
                        'public_metrics': tweet.public_metrics,
                        'is_reply': tweet.in_reply_to_user_id is not None,
                        'is_retweet': tweet.text.startswith('RT @'),
                        'referenced_tweets': tweet.referenced_tweets or []
                    }
                    tweets.append(tweet_data)
                    
                    if len(tweets) >= max_results:
                        break
            
            if len(tweets) >= max_results:
                break
                
    except tweepy.TooManyRequests:
        print(f"Rate limit reached. Retrieved {len(tweets)} tweets so far.")
    except tweepy.Unauthorized:
        raise ValueError("Unauthorized access. Check your bearer token and user permissions.")
    
    return tweets

def save_tweets_to_file(tweets, filename='tweets.json'):
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tweets, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(tweets)} tweets to {filename}")

def load_tweets_from_file(filename='tweets.json'):
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return []