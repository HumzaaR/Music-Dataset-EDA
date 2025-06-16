import numpy as np
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def preprocess_classification(df):
    """Preprocess data for classification: select features, split, scale."""
    features = ['danceability', 'energy', 'duration_ms', 'loudness', 'speechiness', 
                'acousticness', 'instrumentalness', 'valence', 'tempo', 'explicit']
    X = df[features].copy()
    y = (df['popularity'] >= 50).astype(int)
    X['duration_ms'] = X['duration_ms'] / 1000
    X['explicit'] = X['explicit'].astype(int)
    medians = X.median()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, scaler, medians

def train_evaluate_model(X_train, X_test, y_train, y_test):
    """Train and evaluate Logistic Regression model."""
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    return model, accuracy, f1, cm, report_df

def display_model_results(accuracy, f1, cm, report_df):
    """Display model evaluation results in Streamlit."""
    st.write(f"**Accuracy**: {accuracy:.2f} (fraction of correct predictions)")
    st.write(f"**F1-score**: {f1:.2f} (balances precision and recall)")
    st.write("**Confusion Matrix**:")
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    st.pyplot(fig)
    plt.close(fig)
    st.write("True Negatives (top-left), False Positives (top-right), False Negatives (bottom-left), True Positives (bottom-right).")
    st.write("**Classification Report**:")
    st.table(report_df)
    st.write("Shows precision, recall, F1-score per class (0: Not Popular, 1: Popular).")

def predict_popularity(model, scaler, medians):
    """Runtime predictions with simplified user inputs."""
    st.subheader("Predict Popular vs. Not Popular")
    st.write("Enter values for 5 key features to predict if a track is popular (other features use median values):")
    danceability = st.slider("Danceability (0–1): How suitable for dancing (0 = least, 1 = most)", 0.0, 1.0, 0.5)
    energy = st.slider("Energy (0–1): How intense/energetic (0 = calm, 1 = high-energy)", 0.0, 1.0, 0.5)
    loudness = st.slider("Loudness (dB): How loud the track is (-60 = quiet, 0 = loud)", -60.0, 0.0, -10.0)
    valence = st.slider("Valence (0–1): How happy the track feels (0 = sad, 1 = happy)", 0.0, 1.0, 0.5)
    explicit = st.selectbox("Explicit: Contains explicit lyrics (0 = No, 1 = Yes)", [0, 1], index=0)
    input_data = np.array([[danceability, energy, medians['duration_ms'], loudness, 
                            medians['speechiness'], medians['acousticness'], 
                            medians['instrumentalness'], valence, medians['tempo'], explicit]])
    input_data_scaled = scaler.transform(input_data)
    pred = model.predict(input_data_scaled)[0]
    pred_label = "Popular" if pred == 1 else "Not Popular"
    st.write(f"**Predicted Class**: {pred_label}")
    return pred_label