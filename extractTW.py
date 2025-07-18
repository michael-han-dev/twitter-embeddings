from twscrape import API
import asyncio, os, json, re
from twscrape.logger import set_log_level
from dotenv import load_dotenv

load_dotenv()

auth_token = os.getenv("TWITTER_AUTH_TOKEN")
ct0 = os.getenv("TWITTER_CT0")

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


async def main():
    cookies = f"auth_token={auth_token}; ct0={ct0}"
    api = await setup_twscrape(cookies)
    user = await api.user_by_login("michaelyhan_")

    print(f"Scraping tweets for @{user.username} ({user.id})")

    tweets_gen = api.user_tweets(user.id, limit=300)
    tweets = [t async for t in tweets_gen]
    tweet_data=[
        {
            "id":t.id,
            "text": clean(t.rawContent),
            "created_at": t.date.isoformat(),
            "username": t.user.username
        }
        for t in tweets
    ]

    with open("tweets.json", "w") as f:
        json.dump(tweet_data, f, indent=2)

    print(f"Collected {len(tweet_data)} tweets.")

if __name__ == "__main__":
    asyncio.run(main())