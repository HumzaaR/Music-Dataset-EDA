import streamlit as st
import matplotlib.pyplot as plt
from utils import load_data, clean_data
from edas import (eda_1_missing_values, eda_2_data_types, eda_3_summary_stats, 
                  eda_4_outliers, eda_5_popularity_dist, eda_6_genre_popularity, 
                  eda_7_danceability_energy, eda_8_duration_dist, 
                  eda_9_explicit_popularity, eda_10_acousticness_genre, 
                  eda_11_tempo_dist, eda_12_loudness_energy, 
                  eda_13_valence_genre, eda_14_instrumentalness_dist)
from classification import (preprocess_classification, train_evaluate_model, 
                           display_model_results, predict_popularity)

st.set_page_config(page_title="Music Dataset EDA", layout="wide")
st.title("Music Dataset Exploratory Data Analysis")
st.write("**Introduction**")
st.write("This project analyzes a Spotify music dataset for the Introduction to Data Science course.")

# Data Loading
st.header("Data Loading")
st.write("Loading dataset...")
df, load_message = load_data()
if df is None:
    st.error(load_message)
    st.stop()
st.write(load_message)
st.write("First 5 rows:")
st.write(df.head())

# Data Cleaning
st.header("Data Cleaning")
st.write("Cleaning: Remove missing values, convert 'explicit' to boolean, ensure numerical columns.")
df, missing_before, missing_after = clean_data(df)
st.write(f"Missing values before: {missing_before}")
st.write(f"Missing values after: {missing_after}")

# Exploratory Data Analysis
st.header("Exploratory Data Analysis")
st.write("Below are 14 EDAs with visualizations and insights.")

# EDA 1
st.subheader("1. Missing Value Analysis")
st.write("**Purpose**: Check for missing data.")
st.write("**Why Useful**: Ensures data quality.")
missing, insights = eda_1_missing_values(df)
st.table(missing)
st.write(f"**Insights**: {insights}")

# EDA 2
st.subheader("2. Data Types and Unique Values")
st.write("**Purpose**: Show data types and unique counts.")
st.write("**Why Useful**: Understands dataset structure.")
dtypes_unique, insights = eda_2_data_types(df)
st.table(dtypes_unique)
st.write(f"**Insights**: {insights}")

# EDA 3
st.subheader("3. Summary Statistics")
st.write("**Purpose**: Show mean, median, etc. for numerical features.")
st.write("**Why Useful**: Summarizes data spread.")
summary, insights = eda_3_summary_stats(df)
st.table(summary)
st.write(f"**Insights**: {insights}")

# EDA 4
st.subheader("4. Outlier Detection")
st.write("**Purpose**: Find outliers in numerical features.")
st.write("**Why Useful**: Outliers may affect analysis.")
fig, insights = eda_4_outliers(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 5
st.subheader("5. Popularity Distribution")
st.write("**Purpose**: Show popularity distribution.")
st.write("**Why Useful**: Shows common popularity ranges.")
fig, insights = eda_5_popularity_dist(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 6
st.subheader("6. Genre Popularity")
st.write("**Purpose**: Compare popularity across top 5 genres.")
st.write("**Why Useful**: Shows popular genres.")
fig, insights = eda_6_genre_popularity(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 7
st.subheader("7. Danceability vs. Energy")
st.write("**Purpose**: Explore danceability vs. energy.")
st.write("**Why Useful**: Checks if danceable tracks are energetic.")
fig, insights = eda_7_danceability_energy(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 8
st.subheader("8. Duration Distribution")
st.write("**Purpose**: Show track duration distribution.")
st.write("**Why Useful**: Identifies typical lengths.")
fig, insights = eda_8_duration_dist(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 9
st.subheader("9. Explicit vs. Non-Explicit Popularity")
st.write("**Purpose**: Compare popularity by explicit content.")
st.write("**Why Useful**: Checks if explicit tracks are popular.")
fig, insights = eda_9_explicit_popularity(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 10
st.subheader("10. Acousticness by Genre")
st.write("**Purpose**: Show average acousticness for top 5 genres.")
st.write("**Why Useful**: Shows acoustic vs. electronic genres.")
fig, insights = eda_10_acousticness_genre(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 11
st.subheader("11. Tempo Distribution")
st.write("**Purpose**: Show tempo distribution.")
st.write("**Why Useful**: Highlights common tempos.")
fig, insights = eda_11_tempo_dist(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 12
st.subheader("12. Loudness vs. Energy")
st.write("**Purpose**: Explore loudness vs. energy.")
st.write("**Why Useful**: Checks if louder tracks are energetic.")
fig, insights = eda_12_loudness_energy(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 13
st.subheader("13. Valence by Genre")
st.write("**Purpose**: Show average valence for top 5 genres.")
st.write("**Why Useful**: Identifies happy/sad genres.")
fig, insights = eda_13_valence_genre(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# EDA 14
st.subheader("14. Instrumentalness Distribution")
st.write("**Purpose**: Show instrumentalness distribution.")
st.write("**Why Useful**: Identifies instrumental vs. vocal tracks.")
fig, insights = eda_14_instrumentalness_dist(df)
st.pyplot(fig)
st.write(f"**Insights**: {insights}")
plt.close(fig)

# Classification Model
st.header("Classification Model")
st.write("**Model**: Logistic Regression to predict if a track is popular (popularity ≥ 50).")
st.write("**Metrics**: Accuracy, F1-score, Confusion Matrix, Classification Report.")
X_train, X_test, y_train, y_test, scaler, medians = preprocess_classification(df)
model, accuracy, f1, cm, report_df = train_evaluate_model(X_train, X_test, y_train, y_test)
display_model_results(accuracy, f1, cm, report_df)
pred_label = predict_popularity(model, scaler, medians)

# Conclusion
st.header("Conclusion")
st.write("**Key Takeaways**:")
st.write("- Many tracks have low popularity (0–20).")
st.write("- Pop genres tend to be more popular.")
st.write("- Danceability and energy, loudness and energy show positive correlations.")
st.write("- Most tracks are 2–5 minutes, vocal-heavy, with tempos 120–140 BPM.")
st.write(f"- Classification model with 10 features achieves accuracy ~{accuracy:.2f}, F1-score ~{f1:.2f}, with simplified 5-input prediction interface.")
st.write("This analysis and model provide insights into music trends with an easy-to-use prediction tool.")
st.write("**Project by**: Hamza R")