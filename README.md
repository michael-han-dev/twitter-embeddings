# Twitter Embeddings Analysis

Analyze and visualize your Twitter activity using semantic embeddings, clustering, and similarity search. See what topics you've tweeted about and compare it to others. Powered by [@chromadb](https://trychroma.com/)

## Features

- **Tweet Scraping**: Extract tweets using [twscrape](https://github.com/vladkens/twscrape)
- **Semantic Embeddings**: Convert tweets to vectors using local model (all-mpnet-base-v2 via sentence transformers library) or OpenAI (text-embedding-3-small)
- **Clustering**: Group and visualize similar tweets using UMAP + HDBSCAN
- **Interactive Visualization**: 2D/3D plots with animation option
- **Semantic Search**: Find tweets by meaning

## Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo-url>
cd twitter-embeddings
python -m venv env
# source env/bin/activate  # macOS/Linux
# or: env\Scripts\activate  # Windows
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment
```bash
cp .env.example .env
```
Edit `.env` with your API credentials:
- **Twitter/X tokens** (required for scraping)
- **OpenAI API key** (optional, for better embeddings)

### 4. Run the Pipeline
```bash
# Step 1: Scrape tweets
python extractTW.py

# Edit this line (68) for the user and tweet # to scrape (rec 200-300):
new_tweets = await fetch_user_tweets(api, "michaelyhan_", limit=250)

# Step 2: Create embeddings and cluster
python cluster.py

# Step 3: Search your tweets
python search.py
```

## Usage Guide

### Tweet Extraction (`extractTW.py`)
- Scrapes tweets from specified users
- Removes URLs, mentions, hashtags
- Saves to `tweets.json`
- Handles duplicate detection

### Clustering Analysis (`cluster.py`)
- Choose embedding provider (local or OpenAI)
- Select users to analyze (single or compare two)
- Adjust clustering parameters
- Visualize in 2D/3D with animations

**Visualization Controls:**
- `t` - Toggle between representative tweets and keywords
- `r` - Show/hide all tweet text

### Semantic Search (`search.py`)
- Search across all tweets or filter by user
- Find semantically similar content
- Works with both local and OpenAI embeddings

## Configuration

### Twitter API Setup
1. Get your browser cookies from Twitter/X
2. Extract `auth_token` and `ct0` values
3. Add to `.env` file

### OpenAI Setup (Optional)
1. Get API key from OpenAI
2. Add `OPENAI_API_KEY` to `.env`
3. Choose "OpenAI API" option when running `cluster.py`

## Troubleshooting

**Import Errors**: Make sure virtual environment is activated and requirements installed

**Embedding Dimension Mismatch**: Collections created with different embedding providers can't be mixed

**No Collections Found**: Run `cluster.py` first to create embeddings before using `search.py`

**Twitter Scraping Issues**: Check that your browser cookies are valid and up-to-date
