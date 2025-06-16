import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def eda_1_missing_values(df):
    """EDA 1: Missing Value Analysis."""
    missing = df.isna().sum().to_frame(name="Missing Values")
    insights = "No missing values after cleaning."
    return missing, insights

def eda_2_data_types(df):
    """EDA 2: Data Types and Unique Values."""
    dtypes_unique = pd.DataFrame({
        "Data Type": df.dtypes,
        "Unique Values": df.nunique()
    })
    insights = "Numerical features are float; track_genre has many unique values."
    return dtypes_unique, insights

def eda_3_summary_stats(df):
    """EDA 3: Summary Statistics."""
    num_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                'valence', 'tempo']
    summary = df[num_cols].describe().T
    insights = "Popularity is skewed; duration_ms has outliers."
    return summary, insights

def eda_4_outliers(df):
    """EDA 4: Outlier Detection."""
    num_cols = ['popularity', 'duration_ms', 'danceability', 'energy', 'loudness', 
                'speechiness', 'acousticness', 'instrumentalness', 'liveness', 
                'valence', 'tempo']
    fig, ax = plt.subplots(figsize=(10, 6))
    df[num_cols].boxplot(ax=ax)
    plt.xticks(rotation=45)
    plt.title("Box Plot for Numerical Features")
    insights = "Popularity, duration_ms, loudness have outliers."
    return fig, insights

def eda_5_popularity_dist(df):
    """EDA 5: Popularity Distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(df['popularity'], bins=10, edgecolor='black')
    plt.xlabel('Popularity (0–100)')
    plt.ylabel('Number of Tracks')
    plt.title('Popularity Distribution')
    insights = "Many tracks have low popularity (0–20)."
    return fig, insights

def eda_6_genre_popularity(df):
    """EDA 6: Genre Popularity."""
    top_genres = df['track_genre'].value_counts().head(5).index
    df_top = df[df['track_genre'].isin(top_genres)]
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='track_genre', y='popularity', data=df_top)
    plt.xlabel('Genre')
    plt.ylabel('Popularity')
    plt.title('Popularity by Top 5 Genres')
    plt.xticks(rotation=45)
    insights = "Pop genres may be more popular."
    return fig, insights

def eda_7_danceability_energy(df):
    """EDA 7: Danceability vs. Energy."""
    df_sample = df.sample(1000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(df_sample['danceability'], df_sample['energy'], alpha=0.5)
    plt.xlabel('Danceability')
    plt.ylabel('Energy')
    plt.title('Danceability vs. Energy')
    insights = "Positive correlation; dance music clusters high."
    return fig, insights

def eda_8_duration_dist(df):
    """EDA 8: Duration Distribution."""
    durations = df['duration_ms'] / 1000
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(durations, bins=10, edgecolor='black')
    plt.xlabel('Duration (seconds)')
    plt.ylabel('Number of Tracks')
    plt.title('Duration Distribution')
    insights = "Most tracks 2–5 minutes."
    return fig, insights

def eda_9_explicit_popularity(df):
    """EDA 9: Explicit vs. Non-Explicit Popularity."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(x='explicit', y='popularity', data=df)
    plt.xlabel('Explicit')
    plt.ylabel('Popularity')
    plt.title('Popularity: Explicit vs. Non-Explicit')
    insights = "Explicit tracks may be more popular in some genres."
    return fig, insights

def eda_10_acousticness_genre(df):
    """EDA 10: Acousticness by Genre."""
    top_genres = df['track_genre'].value_counts().head(5).index
    df_top = df[df['track_genre'].isin(top_genres)]
    acousticness = df_top.groupby('track_genre')['acousticness'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    acousticness.plot(kind='bar', color='teal')
    plt.xlabel('Genre')
    plt.ylabel('Average Acousticness')
    plt.title('Acousticness by Genre')
    plt.xticks(rotation=45)
    insights = "Folk/classical high; electronic low."
    return fig, insights

def eda_11_tempo_dist(df):
    """EDA 11: Tempo Distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.kdeplot(df['tempo'], fill=True)
    plt.xlabel('Tempo (BPM)')
    plt.ylabel('Density')
    plt.title('Tempo Distribution')
    insights = "Peaks at 120–140 BPM."
    return fig, insights

def eda_12_loudness_energy(df):
    """EDA 12: Loudness vs. Energy."""
    df_sample = df.sample(1000, random_state=42)
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.scatter(df_sample['loudness'], df_sample['energy'], alpha=0.5)
    plt.xlabel('Loudness (dB)')
    plt.ylabel('Energy')
    plt.title('Loudness vs. Energy')
    insights = "Strong positive correlation."
    return fig, insights

def eda_13_valence_genre(df):
    """EDA 13: Valence by Genre."""
    top_genres = df['track_genre'].value_counts().head(5).index
    df_top = df[df['track_genre'].isin(top_genres)]
    valence = df_top.groupby('track_genre')['valence'].mean()
    fig, ax = plt.subplots(figsize=(10, 6))
    valence.plot(kind='bar', color='purple')
    plt.xlabel('Genre')
    plt.ylabel('Average Valence')
    plt.title('Valence by Genre')
    plt.xticks(rotation=45)
    insights = "Pop/dance high; alternative low."
    return fig, insights

def eda_14_instrumentalness_dist(df):
    """EDA 14: Instrumentalness Distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.hist(df['instrumentalness'], bins=10, edgecolor='black')
    plt.xlabel('Instrumentalness')
    plt.ylabel('Number of Tracks')
    plt.title('Instrumentalness Distribution')
    insights = "Most tracks are vocal."
    return fig, insights