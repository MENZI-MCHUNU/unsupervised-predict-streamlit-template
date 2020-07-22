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
movies_df = pd.read_csv('resources/data/movies.csv',sep = ',',delimiter=',')
ratings_df = pd.read_csv('resources/data/ratings.csv')
ratings_df.drop(['timestamp'], axis=1,inplace=True)

# We make use of an SVD model trained on a subset of the MovieLens 10k dataset.
model=pickle.load(open('resources/models/SVD.pkl', 'rb'))

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

    # Build an algorithm, and train it.
    algo = KNNBasic()
    algo.train(a_train)
    predictions = []
    for ui in a_train.all_users():
        predictions.append(algo.predict(iid=item_id,uid=ui, verbose = True))
    return predictions

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
    #movie_list = movie_list[0]
    #indices = pd.Series(movies_df['title'])
    #movie_ids = pred_movies(movie_list)
    #df_init_users = ratings_df[ratings_df['userId']==movie_ids[0]]
    #for i in movie_ids :
    #    df_init_users=df_init_users.append(ratings_df[ratings_df['userId']==i])

    #util_matrix = ratings_df.pivot_table(index=['userId'],
    #                                   columns=['title'],
    #                                   values='rating')    
    # Normalize each row (a given user's ratings) of the utility matrix
    #util_matrix_norm = util_matrix.apply(lambda x: (x-np.mean(x))/(np.max(x)-np.min(x)), axis=1)
    # Fill Nan values with 0's, transpose matrix, and drop users with no ratings
    #util_matrix_norm.fillna(0, inplace=True)
    #util_matrix_norm = util_matrix_norm.T
    #util_matrix_norm = util_matrix_norm.loc[:, (util_matrix_norm != 0).any(axis=0)]
    # Save the utility matrix in scipy's sparse matrix format
    #util_matrix_sparse = sp.sparse.csr_matrix(util_matrix_norm.values)
    # Compute the similarity matrix using the cosine similarity metric
    #user_similarity = cosine_similarity(util_matrix_sparse.T, util_matrix_sparse.T)
    # Save the matrix as a dataframe to allow for easier indexing  
    #user_sim_df = pd.DataFrame(user_similarity,
    #                            index = util_matrix_norm.columns,
    #                            columns = util_matrix_norm.columns)    
    # Getting the cosine similarity matrix
    #cosine_sim = cosine_similarity(np.array(df_init_users), np.array(df_init_users))
    #idx_1 = indices[indices == movie_list[0]].index[0]
    #idx_2 = indices[indices == movie_list[1]].index[0]
    #idx_3 = indices[indices == movie_list[2]].index[0]
    # Creating a Series with the similarity scores in descending order
    #rank_1 = cosine_sim[idx_1]
    #rank_2 = cosine_sim[idx_2]
    #rank_3 = cosine_sim[idx_3]
    #rank_1 = user_similarity[idx_1]
    #rank_2 = user_similarity[idx_2]
    #rank_3 = user_similarity[idx_3]
    # Calculating the scores
    #score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    #score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    #score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
     # Appending the names of movies
    #listings = score_series_1.append(score_series_1).append(score_series_3).sort_values(ascending = False)
    #recommended_movies = []
    # Choose top 50
    #top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    #top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    #for i in top_indexes[:top_n]:
    #    recommended_movies.append(list(movies_df['title'])[i])
    #return recommended_movies
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
    
    def fuzzy_matching(mapper, fav_movie, verbose=True):
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
            ratio = fuzz.ratio(title.lower(), fav_movie[0].lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
                
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), fav_movie[1].lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))
        # get match
        for title, idx in mapper.items():
            ratio = fuzz.ratio(title.lower(), fav_movie[2].lower())
            if ratio >= 60:
                match_tuple.append((title, idx, ratio))                
        # sort
        match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
        if not match_tuple:
            print('Oops! No match is found')
            return
        if verbose:
            #print('Found possible matches in our database: {0}\n'.format([x[0] for x in match_tuple]))
            list_ = [x[0] for x in match_tuple]
        return list_#match_tuple[0][1]    
    
    # Store movie names
    recommended_movies = []
    # get input movie index
    #print('You have input movie:', fav_movie)
    idx = fuzzy_matching(movie_to_idx, fav_movie, verbose=True)
    listings = pd.Series(idx)
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[fav_movie[0],fav_movie[1],fav_movie[2]]) 
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies_df['title'])[i])    
    return recommended_movies