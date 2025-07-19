from twscrape import API
import asyncio, os, json, re
from twscrape.logger import set_log_level
from dotenv import load_dotenv

load_dotenv()

auth_token = os.getenv("TWITTER_AUTH_TOKEN")
ct0 = os.getenv("TWITTER_CT0")

#create an account to scrape with
async def setup_twscrape(cookies: str, db_path: str= "accounts.db") -> API:
    set_log_level("INFO")
    api = API(db_path)

    await api.pool.add_account(
    username="dummy_user",         
    password="dummy_pass",         
    email="dummy@email.com",       
    email_password="dummy_email_pw",  
    cookies=cookies
    )
    await api.pool.login_all()
    return api

def clean(txt: str) -> str:
    txt = re.sub(r"http\S+", "", txt)
    txt = re.sub(r"@\w+", "", txt)
    txt = txt.replace("#", "")
    return txt.strip()

def load_tweets(file= 'tweets.json'):
    try:
        with open(file, 'r', encoding='utf-8') as f:
            json.load(f)
    except FileNotFoundError:
        return []
async def fetch_user_tweets(api: API, username: str, limit: int=500) -> list[dict]:
    #remove duplicate tweets scraped and scrape given username
    user = await api.user_by_login(username)
    seen = set()
    tweet_data = []

    #scrape only user's tweets and not replies
    async for t in api.user_tweets(user.id, limit):
        if t.id in seen:
            continue
        seen.add(t.id)

        #choose the tweet data to save, username in case user wants to compare more than 1 person's tweets for filtering later
        tweet_data.append({
            "id": t.id,
            "text": clean(t.rawContent),
            "created_at": t.date.isoformat(),
            "username": user.username
        })
    return tweet_data

async def main():
    cookies = f"auth_token={auth_token}; ct0={ct0}"
    api = await setup_twscrape(cookies)

    tweet_data = await fetch_user_tweets(api, "michaelyhan_", limit=1000)

    with open("tweets.json", "w") as f:
        json.dump(tweet_data, f, indent=2)

    print(f"Collected {len(tweet_data)} tweets.")

if __name__ == "__main__":
    asyncio.run(main())