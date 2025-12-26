import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def build_user_item_matrix(ratings):
    rating_df = ratings.drop(columns=["timestamp"])
    user_item_matrix = rating_df.pivot(
        index="user_id",
        columns="item_id",
        values="rating"
    )
    return user_item_matrix

def compute_item_similarity(user_item_matrix):
    item_user_matrix = user_item_matrix.T.fillna(0)
    similarity = cosine_similarity(item_user_matrix)
    return pd.DataFrame(
        similarity,
        index=item_user_matrix.index,
        columns=item_user_matrix.index
    )

def recommend_items_item_based(user_id, user_item_matrix, item_similarity, top_k=5):
    user_ratings = user_item_matrix.loc[user_id].dropna()
    scores = {}

    for item, rating in user_ratings.items():
        for candidate, sim in item_similarity[item].items():
            if pd.isna(user_item_matrix.loc[user_id, candidate]):
                scores[candidate] = scores.get(candidate, 0) + sim * rating

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_k]]
