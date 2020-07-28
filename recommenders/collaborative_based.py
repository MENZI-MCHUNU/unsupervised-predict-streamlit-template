"""

    Collaborative-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `collab_model` !!

    You must however change its contents (i.e. add your own collaborative
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline collaborative
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import pandas as pd
import numpy as np
import pickle
import copy
from surprise import Reader, Dataset
from surprise import SVD, NormalPredictor, BaselineOnly, KNNBasic, NMF
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix
# Importing data
from fuzzywuzzy import fuzz
from sklearn.neighbors import NearestNeighbors
movies_df = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/movies.csv',sep = ',',delimiter=',')
ratings_df = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/train.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
#model=pickle.load(open('/resources/models/SVD.pkl', 'rb'))

def prediction_item(item_id):
    """Map a given favourite movie to users within the
       MovieLens dataset with the same preference.

    Parameters
    ----------
    item_id : int
        A MovieLens Movie ID.

    Returns
    -------
    list
        User IDs of users with similar high ratings for the given movie.

    """
    # Data preprosessing
    reader = Reader(rating_scale=(0, 5))
    load_df = Dataset.load_from_df(ratings_df,reader)
    a_train = load_df.build_full_trainset()


    #for ui in a_train.all_users():
        #predictions.append(model.predict(iid=item_id,uid=ui, verbose = True))
    return a_train #predictions

def pred_movies(movie_list):
    """Maps the given favourite movies selected within the app to corresponding
    users within the MovieLens dataset.

    Parameters
    ----------
    movie_list : list
        Three favourite movies selected by the app user.

    Returns
    -------
    list
        User-ID's of users with similar high ratings for each movie.

    """
    # Store the id of users
    id_store=[]
    # For each movie selected by a user of the app,
    # predict a corresponding user within the dataset with the highest rating
    for i in movie_list:
        predictions = prediction_item(item_id = i)
        predictions.sort(key=lambda x: x.est, reverse=True)
        # Take the top 10 user id's from each movie with highest rankings
        for pred in predictions[:10]:
            id_store.append(pred.uid)
    # Return a list of user id's
    return id_store

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.  
def collab_model(movie_list,top_n=10):
    """Performs Collaborative filtering based upon a list of movies supplied
       by the app user.

    Parameters
    ----------
    movie_list : list (str)
        Favorite movies chosen by the app user.
    top_n : type
        Number of top recommendations to return to the user.

    Returns
    -------
    list (str)
        Titles of the top-n movie recommendations to the user.

    """
    
    df_movies_cnt = pd.DataFrame(ratings_df.groupby('movieId').size(), columns=['count'])
    popularity_thres = 50
    popular_movies = list(set(df_movies_cnt.query('count >= @popularity_thres').index))
    df_ratings_drop_movies = ratings_df[ratings_df.movieId.isin(popular_movies)]
    
    # get number of ratings given by every user
    df_users_cnt = pd.DataFrame(df_ratings_drop_movies.groupby('userId').size(), columns=['count'])
    ratings_thres = 50
    active_users = list(set(df_users_cnt.query('count >= @ratings_thres').index))
    df_ratings_drop_users = df_ratings_drop_movies[df_ratings_drop_movies.userId.isin(active_users)]
    # pivot and create movie-user matrix
    movie_user_mat = df_ratings_drop_users.pivot(index='movieId', columns='userId', values='rating').fillna(0)
    # create mapper from movie title to index
    movie_to_idx = {
        movie: i for i, movie in 
        enumerate(list(movies_df.set_index('movieId').loc[movie_user_mat.index].title))
    }
    # transform matrix to scipy sparse matrix
    movie_user_mat_sparse = csr_matrix(movie_user_mat.values)
    
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)
    # fit
    model_knn.fit(movie_user_mat_sparse)
    # fit
    #model_knn.fit(data)
    
    def fuzzy_matching(mapper, movie_list, verbose=True):
        """
        return the closest match via fuzzy ratio. If no match found, return None
    
        Parameters
        ----------    
        mapper: dict, map movie title name to index of the movie in data

        fav_movie: str, name of user input movie
    
        verbose: bool, print log if True

        Return
        ------
        index of the closest match
        """
        match_tuple = []
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), movie_list[0].lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
                
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), movie_list[1].lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), movie_list[2].lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))                
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
            return
        if verbose:
            list_ = [x[0] for x in match_tuple]
        return list_  
    
    # Store movie names
    recommended_movies = []
    idx = fuzzy_matching(movie_to_idx, movie_list, verbose=True)
    unwanted = [movie_list[0],movie_list[1],movie_list[2]]    
    list1 = [ele for ele in idx if ele not in unwanted]
    return list1[:top_n]