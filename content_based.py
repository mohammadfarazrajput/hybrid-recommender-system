from sklearn.metrics.pairwise import cosine_similarity

def content_based(movie_index, movie_data,titles, top_n_recommedation = 5):
    similar_matrix = cosine_similarity(movie_data.iloc[[movie_index]],movie_data)  
    indexed_scores = list(enumerate(similar_matrix[0]))
    sorted_recommedations = sorted(
        indexed_scores,
        key=lambda x: x[1],
        reverse=True
    )[1:top_n_recommedation + 1]   # skip self

    recommedation_index = [idx for idx, score in sorted_recommedations]
    final_recommendation = titles.iloc[recommedation_index].reset_index(drop =True)
    print("The Recommedations are:\n")
    return final_recommendation