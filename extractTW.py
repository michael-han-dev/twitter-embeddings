from twscrape import API
import asyncio
from twscrape.logger import set_log_level
from dotenv import load_dotenv
import os

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
    return api

async def main():
    cookies = f"auth_token={auth_token}; ct0={ct0}"
    api = await setup_twscrape(cookies)
    user = await api.user_by_login("xdevelopers")
    print(user.dict())

if __name__ == "__main__":
    asyncio.run(main())