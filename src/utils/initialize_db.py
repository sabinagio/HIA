import chromadb
from chromadb.errors import InvalidCollectionException
from chromadb.utils import embedding_functions

import pandas as pd
import numpy as np
import json

offers = pd.read_csv("data/Offers Clean.csv")

def initialize_vectorstore():
    """Initialize and return Chroma vectorstore with embeddings"""
    # Initialize persistent Chroma client
    client = chromadb.PersistentClient(path="./chroma_db")

    # client.delete_collection("test_collection")

    # Create or get existing collection
    try:
        collection = client.get_collection(
            name="test_collection",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        print("Collection obtained.")
    except (ValueError, InvalidCollectionException):  # Collection doesn't exist
        collection = client.create_collection(
            name="test_collection",
            embedding_function=embedding_functions.DefaultEmbeddingFunction()
        )
        print("Collection created.")

    documents = offers['offer_edited'].to_list()

    # Add metadata to each document
    meta_cols = ['subdomain', 'icon_url', 'link', 'address', 'date_added']
    metadatas = offers[meta_cols].to_dict(orient='records')
    comp_metadatas = offers[["email", "phone_number", "opening_hours_weekday", "opening_hours_weekend"]].to_dict(orient='records')
    for metadata, comp_metadata in zip(metadatas, comp_metadatas):
        metadata['contact'] = json.dumps({"weekday": comp_metadata["opening_hours_weekday"], "weekend": comp_metadata["opening_hours_weekend"]})
        metadata['opening_hours'] = json.dumps({"email": comp_metadata["email"], "phone": comp_metadata["phone_number"]})
        metadata['source'] = np.nan
        metadata['category'] = "TBD"

    # Add documents to collection
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=[f"doc_{i}" for i in range(len(documents))]
    )

    return collection

initialize_vectorstore()
