import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import EDA_functions

def select_an_images (id,df_images) : 
    url = df_images[df_images['id']== id]["images"]
    first_non_empty = url[url.apply(lambda x: len(x) > 0)].iloc[0]  # Get the first non-empty value
    url = first_non_empty[0].get("small_image_url")
    print(f"URL de l'image de l'objet {url}")
    EDA_functions.show_image(url)


def Csimilarity_user_recommendation(df_recommendation, select_userid , df_images, k_similars = 10 , user_similarity_threshold = 0.3, nb_recommendation = 7) : 

    print(f"The number of unique products is : {df_recommendation.id.nunique()}")
    print(f"The number of unique users is : {df_recommendation.user.nunique()} \n")
    df_recommendation_pivot = df_recommendation.pivot(index='user', columns='id', values='rating')
    df_recommendation_pivot = df_recommendation_pivot.fillna(0)

    similarity_matrix = cosine_similarity(df_recommendation_pivot)
    df_similarity_matrix = pd.DataFrame(similarity_matrix, index= df_recommendation_pivot.index, columns= df_recommendation_pivot.index)

    # Affichage  des produits que l'utilisateur selectioné a noté dans le passée :
    products_rated_by_a = df_recommendation_pivot.loc[select_userid, df_recommendation_pivot.loc[select_userid, :] != 0].index
    print ("   v v v v v Here is images of the product our user bought in the past :    v v v v v")
    for id in products_rated_by_a : 
        select_an_images(id, df_images)

    # top k similar users
    similar_users = df_similarity_matrix[df_similarity_matrix[select_userid]>user_similarity_threshold][select_userid].sort_values(ascending=False)[:k_similars]
    similar_users_df = similar_users.to_frame(name='similarity')

    print (f"The similar (with similarity bigger than {user_similarity_threshold} of cosine similarity) users to user {select_userid} are  : ")

    for index, row in similar_users_df.iterrows():
        similarity = row['similarity'] 
        print(f"User: {index}, Similarity: {similarity}")

    bought_products = df_recommendation_pivot.loc[df_recommendation_pivot.index== select_userid, df_recommendation_pivot.loc[select_userid,:]>=3]

    not_rated_by_user = df_recommendation_pivot[(df_recommendation_pivot.loc[select_userid, :] == 0).index[(df_recommendation_pivot.loc[select_userid, :] == 0).values]]
    not_bought_products = df_recommendation_pivot.loc[
    df_recommendation_pivot.index != select_userid,  # Exclude selected user
    not_rated_by_user.columns  # products not rated by the selected user
    ]


    # Select only the rows of not_bought_products for similar users
    similar_user_ids = similar_users_df.drop(select_userid).index
    filtered_not_bought_products = not_bought_products.loc[similar_user_ids]

    # Normalize weights
    weights = similar_users_df.drop(select_userid) / similar_users_df.drop(select_userid).sum()

    # Compute the weighted averages
    weighted_averages = filtered_not_bought_products.T.dot(weights.to_numpy())

    weighted_averages = pd.DataFrame(weighted_averages)#.sort_values(by=0, ascending=False))
    weighted_averages.columns = ['weighted_averages']

    weighted_averages.sort_values(by="weighted_averages", ascending = False)
    top_recommendation = weighted_averages.sort_values(by="weighted_averages", ascending = False).head(nb_recommendation)

    i = 1
    for index, rows in top_recommendation.iterrows(): 
        print (f"*************** Recommendation number {i} *************** : \n Product id : {index}, associated weight : {rows['weighted_averages']}")
        select_an_images(index,df_images)
        i += 1

    return top_recommendation