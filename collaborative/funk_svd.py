import numpy as np

def create_id_mappings(ratings):
    user_map = {u: i for i, u in enumerate(ratings["user_id"].unique())}
    item_map = {m: i for i, m in enumerate(ratings["item_id"].unique())}
    return user_map, item_map, len(user_map), len(item_map)

def create_latent_matrix(user_len, item_len, k=25):
    np.random.seed(42)
    U = np.random.normal(0, 0.05, (user_len, k))
    I = np.random.normal(0, 0.05, (item_len, k))
    return U, I

def gradient_update(U, user_map, I, item_map, ratings, alpha=0.01, reg=0.01, epoches=50):
    global_mean = ratings["rating"].mean()
    user_bias = np.zeros(len(user_map))
    item_bias = np.zeros(len(item_map))

    for _ in range(epoches):
        for u, i, r in ratings[["user_id", "item_id", "rating"]].values:
            u = user_map[u]
            i = item_map[i]

            pred = global_mean + user_bias[u] + item_bias[i] + np.dot(U[u], I[i])
            err = r - pred

            user_bias[u] += alpha * (err - reg * user_bias[u])
            item_bias[i] += alpha * (err - reg * item_bias[i])

            old_u = U[u].copy()
            U[u] += alpha * (err * I[i] - reg * U[u])
            I[i] += alpha * (err * old_u - reg * I[i])

    return U, I, user_bias, item_bias, global_mean

def funk_svd_recommend(
    U, user_map, I, item_map,
    user_id, ratings,
    user_bias, item_bias, global_mean,
    top_k=5
):
    u = user_map[user_id]
    scores = {}

    for item_id, i in item_map.items():
        if not ((ratings["user_id"] == user_id) & (ratings["item_id"] == item_id)).any():
            score = global_mean + user_bias[u] + item_bias[i] + np.dot(U[u], I[i])
            scores[item_id] = score

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [item for item, _ in ranked[:top_k]]
