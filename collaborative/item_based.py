import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity 

def build_user_item_matrix(ratings):
    rating_df = ratings.drop(columns = ['timestamp'])    
    user_item_matrix = rating_df.pivot(index = 'user_id', columns= ['item_id'], values = ['rating'])
    print(f"The shape of the ratings matrix is {user_item_matrix.shape}")
    return user_item_matrix

def compute_item_similarity(user_item_matrix):
    item_user_matrix = user_item_matrix.T
    print("The shape of the transpose matrix is {item_user_matrix.shape}")
    item_similarity  = cosine_similarity(item_user_matrix.fillna(0))
    item_similarity = pd.DataFrame(item_similarity, columns = user_item_matrix.columns, index= user_item_matrix.columns)
    return item_similarity 

def recommend_items_item_based(user_id, rating_df, item_similarity, top_k=5):
    reviewed_movies = rating_df.loc[user_id].dropna()
    score = {}
    for item, rating in reviewed_movies.items():
        candidate_movies_score = item_similarity[item]
        for candidate_movies, similarity in candidate_movies_score.items():
            if pd.isna(rating_df.loc[user_id,candidate_movies]):
                contribution  = similarity*rating     
                score[candidate_movies]=score.get(candidate_movies,0)+ contribution
    ranked_items  = sorted(score.items(), key = lambda x: x[1], reverse= True)
    return [item for item, score in ranked_items[1:top_k+1]]