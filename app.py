import streamlit as st
import numpy as np
import pickle
import pandas as pd

from data_loader import MoviesLensLoader
from collaborative.item_based import (
    build_user_item_matrix,
    compute_item_similarity,
    recommend_items_item_based
)
from collaborative.funk_svd import funk_svd_recommend
from hybrid_rec import hybrid_recommendation
from sklearn.feature_extraction.text import TfidfVectorizer

# ----------------------------
# Load Data
# ----------------------------
st.set_page_config(page_title="Hybrid Movie Recommender", layout="wide")

st.title("🎬 Hybrid Movie Recommendation System")

loader = MoviesLensLoader()
ratings, items = loader.load_all()

# ----------------------------
# Load Models
# ----------------------------
U = np.load("artifacts/U.npy")
I = np.load("artifacts/I.npy")
user_bias = np.load("artifacts/user_bias.npy")
item_bias = np.load("artifacts/item_bias.npy")
global_mean = float(np.load("artifacts/global_mean.npy"))

with open("artifacts/user_map.pkl", "rb") as f:
    user_map = pickle.load(f)

with open("artifacts/item_map.pkl", "rb") as f:
    item_map = pickle.load(f)

# ----------------------------
# Content Features
# ----------------------------
tf = TfidfVectorizer(stop_words="english")
tf_title = tf.fit_transform(items["title"].str.lower())

tf_df = pd.DataFrame(
    tf_title.toarray(),
    columns=tf.get_feature_names_out()
)

genre_cols = [f"genre_{i}" for i in range(19)]
content_features = pd.concat(
    [tf_df, items[genre_cols].reset_index(drop=True)],
    axis=1
)

# ----------------------------
# Item-Based CF Prep
# ----------------------------
user_item_matrix = build_user_item_matrix(ratings)
item_similarity_df = compute_item_similarity(user_item_matrix)

# ----------------------------
# Sidebar Inputs
# ----------------------------
st.sidebar.header("User Inputs")

user_id = st.sidebar.number_input(
    "Enter user_id",
    min_value=int(ratings["user_id"].min()),
    max_value=int(ratings["user_id"].max()),
    step=1
)

movie_title = st.sidebar.selectbox(
    "Select a movie you liked",
    sorted(items["title"].unique())
)

top_k = st.sidebar.slider("Number of recommendations", 3, 10, 5)

run_btn = st.sidebar.button("Get Recommendations 🚀")

# ----------------------------
# Recommendation Logic
# ----------------------------
if run_btn:
    st.subheader("Results")

    # -------- Item-Based CF --------
    item_cf_ids = recommend_items_item_based(
        user_id,
        user_item_matrix,
        item_similarity_df,
        top_k
    )

    st.markdown("### 🔵 Item-Based Collaborative Filtering")
    st.table(items[items["item_id"].isin(item_cf_ids)][["title"]])

    # -------- FunkSVD --------
    svd_ids = funk_svd_recommend(
        U, user_map, I, item_map,
        user_id, ratings,
        user_bias, item_bias, global_mean,
        top_k
    )

    st.markdown("### 🟢 FunkSVD Recommendations")
    st.table(items[items["item_id"].isin(svd_ids)][["title"]])

    # -------- Hybrid --------
    seed_idx = items[items["title"] == movie_title].index[0]

    hybrid_ids = hybrid_recommendation(
        user_id,
        seed_idx,
        items,
        ratings,
        content_features,
        U, I, user_map, item_map,
        user_bias, item_bias, global_mean,
        top_k=top_k
    )

    st.markdown("### 🔥 Hybrid Recommendations (Final)")
    st.table(items[items["item_id"].isin(hybrid_ids)][["title"]])
