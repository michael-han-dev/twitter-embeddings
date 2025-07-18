import os
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.common.exceptions import TimeoutException
from datetime import datetime
import random
import json
import time

load_dotenv()

def load_cookies(driver, cookies_path):
    with open(cookies_path, "r") as f:
        cookies = json.load(f)
    
    for cookie in cookies:
        cookie.pop("sameSite", None)
        driver.add_cookie(cookie)


def scroll_user_tweets(username: str, scroll_count: int=5, headless: bool=True):
    options = Options()
    driver = webdriver.Chrome(options=options)

    driver.get("https://x.com/")
    load_cookies(driver, "x_cookies.json")
    driver.refresh()

    url = f"https://x.com/{username}"
    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
    )

    articles = driver.find_elements(By.CSS_SELECTOR, "article")
    scrolls = 0
    for i in range(scroll_count):
        prev_count = len(articles)
        r = random.randint(0,18)
        if i % 3 == 0:
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        else:
            scroll_distance = random.randint(800, 1700)
            driver.execute_script(f"window.scrollBy(0, {scroll_distance});")
        print(f'scroll count = {i + 1}')
        try:
            WebDriverWait(driver, 10).until(
                lambda d: len(d.find_elements(By.CSS_SELECTOR, "article")) > prev_count
            )
            time.sleep(random.uniform(2.5, 6))
            scrolls += 1
        except TimeoutException:
            print("No new tweets loaded. Ending scroll early.")
            break

        articles = driver.find_elements(By.CSS_SELECTOR, "article")
    return driver


if __name__ == "__main__":
    username="michaelyhan_"
    driver = scroll_user_tweets(username , scroll_count=10)
    articles = driver.find_elements(By.CSS_SELECTOR, "article")
    print(f"Loaded {len(articles)} tweets.")
    for a in articles:
        print("---")
        print(a.text)

    # Filter real tweets
    tweets = [a for a in articles if username in a.text]
    print(f"Real tweets: {len(tweets)}")
    print(f"Total articles detected: {len(articles)}")
    driver.quit()