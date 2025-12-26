from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def content_based(movie_index, feature_df, top_n=50):
    movie_vec = feature_df.iloc[[movie_index]]
    similarity = cosine_similarity(movie_vec, feature_df)[0]

    scores = {
        idx: score
        for idx, score in enumerate(similarity)
        if idx != movie_index
    }

    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return dict(sorted_scores[:top_n])
