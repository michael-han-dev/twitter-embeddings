import os
from dotenv import load_dotenv
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from datetime import datetime
import json

load_dotenv()

def scroll_user_tweets(username: str, scroll_count: int=5, headless: bool=True):
    options = Options()
    # if headless:
    #     options.add_argument("--headless")
    driver = webdriver.Chrome(options=options)

    url = f"https://x.com/{username}"
    driver.get(url)
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "article"))
    )

    articles = driver.find_elements(By.CSS_SELECTOR, "article")
    for _ in range(scroll_count):
        prev_count = len(articles)
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

        WebDriverWait(driver, 10).until(
            lambda d:len(d.find_elements(By.CSS_SELECTOR, "article")) > prev_count
        )

        articles = driver.find_elements(By.CSS_SELECTOR, "article")
    return driver


if __name__ == "__main__":
    driver = scroll_user_tweets("michaelyhan_", scroll_count=5)
    articles = driver.find_elements(By.CSS_SELECTOR, "article")
    print(f"Loaded {len(articles)} tweets.")
    driver.quit()