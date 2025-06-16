import pandas as pd
from datasets import load_dataset
import os

def load_data():
    """Load dataset from Hugging Face or local CSV."""
    try:
        dataset = load_dataset("maharshipandya/spotify-tracks-dataset", split="train")
        df = pd.DataFrame(dataset)
        return df, "Dataset loaded from Hugging Face."
    except:
        if os.path.exists("dataset.csv"):
            df = pd.read_csv("dataset.csv")
            return df, "Dataset loaded from local 'dataset.csv'."
        else:
            return None, "Dataset not found. Place 'dataset.csv' in the same directory."

def clean_data(df):
    """Clean dataset: remove missing values, convert types."""
    missing_before = df.isna().sum().sum()
    df = df.dropna()
    df['explicit'] = df['explicit'].astype(bool)
    num_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                'valence', 'tempo']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna()
    missing_after = df.isna().sum().sum()
    return df, missing_before, missing_after