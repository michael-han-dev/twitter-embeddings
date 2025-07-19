import numpy as np
import umap
import hdbscan
import json
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation as LDA

import chromadb

client = chromadb.PersistentClient(path="./chroma.db")

def cluster(collection_name="tweets", username: str | None = None):
    collection = client.get_collection(collection_name)
    where_clause = {"username": username} if username else {}

    
