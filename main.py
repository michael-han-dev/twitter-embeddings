from extractTW import setup_twscrape
from dotenv import load_dotenv
import os, json, re

load_dotenv()

auth_token = os.getenv("TWITTER_AUTH_TOKEN")
ct0 = os.getenv("TWITTER_CT0")
