import numpy as np
def create_id_mappings(ratings):
    user_map = {user:idx for idx, user in enumerate(ratings['user_id'].unique())}
    item_map = {item:idx for idx, item in enumerate(ratings['item_id'].unique())}
    return user_map, item_map, len(user_map), len(item_map)
def create_latent_matrix (user_map_len, item_map_len, k =25):
    np.random.seed(42)
    U = np.round(np.random.normal(loc = 0, scale = 0.05, size = (user_map_len,k)), decimals= 2)
    I = np.round(np.random.normal(loc = 0, scale = 0.05, size =(item_map_len, k)), decimals= 2)
    return U, I
def gradient_update(U,user_map,I,item_map,ratings,user_map_len, item_map_len, alpha = 0.01, reg = 0.01, epoches = 100):
    global_mean = ratings['rating'].mean()
    user_bias = np.zeros(user_map_len)
    item_bias = np.zeros(item_map_len)

    for _ in range(epoches):    
        for user,item,rating in ratings[["user_id", "item_id", "rating"]].values:
            mapped_user_id = user_map[user]
            mapped_item_id = item_map[item]   
            predict = global_mean + user_bias[mapped_user_id] + item_bias[mapped_item_id] + np.dot(U[mapped_user_id],I[mapped_item_id])
            error = rating - predict
            user_bias[mapped_user_id] += alpha * (error - reg * user_bias[mapped_user_id])
            item_bias[mapped_item_id] += alpha * (error - reg * item_bias[mapped_item_id])
            old_vector_user =  U[mapped_user_id]
            U[mapped_user_id] += alpha* (error * I[mapped_item_id] - reg * U[mapped_user_id]) 
            I[mapped_item_id] += alpha* ( error * old_vector_user - reg * I[mapped_item_id]) 
    return U, I, user_bias, item_bias, global_mean 
    
def funk_svd_recommend(U,user_map,I,item_map,item_len, current_user, ratings, user_bias, item_bias, global_mean, top_k =5):
    print(f"the user_id  you types is : {current_user}")
    mapped_user_id = user_map[current_user]
    #print(f"the current user id is {current_user}")
    score_list = []
    for values, item_idx in item_map.items() :  
        #print(f"the length of the item_map is {len(item_map)}")
        #print(f"the value for the current item {item_idx} is {I[item_idx-1]}")
        score_list.append((global_mean + user_bias[mapped_user_id] + item_bias[mapped_item_id] + np.dot(U[mapped_user_id],I[mapped_item_id])))
    #print(f"the score_list is updated {score_list}")    
    score_map = {idx : float(score) for idx, score in enumerate(score_list)}  
    rated_movies_by_current_user = ratings[ratings['user_id'] == current_user]['item_id'].to_list()
    for movie_id in rated_movies_by_current_user:
        mapped_item_id = item_map[movie_id]
        score_map.pop(mapped_item_id)
    sorted_index = sorted(score_map.items(), key= lambda x: x[1], reverse = True)    
    top_recommedation_sorted_idx_score = sorted_index[:top_k]
    top_recommedation_sorted_idx = []
    for item_map_id, score in top_recommedation_sorted_idx_score:
        top_recommedation_sorted_idx.append(item_map_id)
    recommend_movie_id = [int(item_id) for item_id, item_map_id in item_map.items() if item_map_id in top_recommedation_sorted_idx]
    return recommend_movie_id  