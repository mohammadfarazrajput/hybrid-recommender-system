import numpy as np
from content_based import content_based

def hybrid_recommendation(
    user_id,
    seed_movie_index,
    items,
    ratings,
    content_features,
    U, I, user_map, item_map,
    user_bias, item_bias, global_mean,
    w_content=0.3,
    w_svd=0.7,
    top_k=5
):
    content_scores = content_based(seed_movie_index, content_features)

    final_scores = {}
    u = user_map[user_id]

    for idx, c_score in content_scores.items():
        item_id = items.iloc[idx]["item_id"]
        if item_id not in item_map:
            continue

        i = item_map[item_id]
        svd_score = global_mean + user_bias[u] + item_bias[i] + np.dot(U[u], I[i])
        svd_norm = (svd_score - 1) / 4

        final_scores[item_id] = w_content * c_score + w_svd * svd_norm

    rated = ratings[ratings["user_id"] == user_id]["item_id"].tolist()
    for r in rated:
        final_scores.pop(r, None)

    ranked = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_k]]
