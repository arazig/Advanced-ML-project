import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import EDA_functions
import matplotlib.pyplot as plt
import requests
from PIL import Image
from io import BytesIO

def select_an_images (id,df_metadata) : 
        url = df_metadata[df_metadata['id']== id]["images"]
        #first_non_empty = url[url.apply(lambda x: len(x) > 0)].iloc[0]  # Get the first non-empty value
        #url = first_non_empty[0].get("small_image_url")
        for element in url : 
            #print(element)
            el = element[0].get("large")
            print(f"URL de l'image de l'objet {el}")
        EDA_functions.show_image(el)


def plot_images( item_id, df_metadata):
    """
    Plot an item for a given item id.
    """
    nom_item = df_metadata[df_metadata['id'] == item_id]["title"].iloc[0]
    image_url = df_metadata[df_metadata['id'] == item_id]['images'].iloc[0][0]['large']
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))

    # Create a plot
    fig, axs = plt.subplots(1, 1, figsize=(2, 2))  # Smaller figure size
    axs.imshow(img)
    axs.set_title(f"Item {item_id}\n Name : {nom_item}", fontsize=10)
    axs.axis('off')
    plt.tight_layout()
    plt.show()


def create_image_grid(id_list, df_metadata, items_per_row=4):
    """
    Create a grid of images given a list of item IDs and metadata.
    
    Args:
        id_list (list): List of item IDs to display.
        df_metadata (DataFrame): DataFrame containing metadata for the items.
        items_per_row (int): Number of images per row.
    """
    num_items = len(id_list)
    num_rows = (num_items + items_per_row - 1) // items_per_row  # Calculate number of rows needed
    
    # Create the figure and subplots
    fig, axs = plt.subplots(num_rows, items_per_row, figsize=(items_per_row * 4, num_rows * 4))
    axs = axs.flatten()  # Flatten the axes for easier indexing

    for i, item_id in enumerate(id_list):
        # Get image URL and title from the metadata
        try:
            nom_item = df_metadata[df_metadata['id'] == item_id]["title"].iloc[0]
            image_url = df_metadata[df_metadata['id'] == item_id]['images'].iloc[0][0]['large']
            
            # Fetch and open the image
            response = requests.get(image_url)
            img = Image.open(BytesIO(response.content))

            # Display the image and title
            axs[i].imshow(img)
            axs[i].set_title(f"{nom_item}", fontsize=8)
            axs[i].axis('off')
        except Exception as e:
            axs[i].axis('off')  # Hide the axis if there's an issue
            print(f"Error loading item {item_id}: {e}")

    # Hide unused subplots
    for j in range(i + 1, len(axs)):
        axs[j].axis('off')

    plt.tight_layout()
    plt.show()



def Csimilarity_user_recommendation(df_recommendation, select_userid , df_metadata, k_similars = 10 , user_similarity_threshold = 0.3, nb_recommendation = 7) : 

    print(f"The number of unique products is : {df_recommendation.id.nunique()}")
    print(f"The number of unique users is : {df_recommendation.user.nunique()} \n")
    df_recommendation_pivot = df_recommendation.pivot(index='user', columns='id', values='rating')
    df_recommendation_pivot = df_recommendation_pivot.fillna(0)

    similarity_matrix = cosine_similarity(df_recommendation_pivot)
    df_similarity_matrix = pd.DataFrame(similarity_matrix, index= df_recommendation_pivot.index, columns= df_recommendation_pivot.index)

    # Affichage  des produits que l'utilisateur selectioné a noté dans le passée :
    products_rated_by_a = df_recommendation_pivot.loc[select_userid, df_recommendation_pivot.loc[select_userid, :] != 0].index
    print ("   v v v v v Here is images of the product our user bought in the past :    v v v v v")
    create_image_grid(products_rated_by_a, df_metadata)

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

    print (f"\n   v v v v v Here is the products recommended for user  {select_userid} :    v v v v v")

    i = 1
    for index, rows in top_recommendation.iterrows(): 
        if rows['weighted_averages'] > 0.001 : 
            print (f"*************** Recommendation number {i} *************** : \n Product id : {index}, associated weight : {rows['weighted_averages']}")
            plot_images(index,df_metadata)
            i += 1

    return top_recommendation