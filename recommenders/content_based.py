"""

    Content-based filtering for item recommendation.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: You are required to extend this baseline algorithm to enable more
    efficient and accurate computation of recommendations.

    !! You must not change the name and signature (arguments) of the
    prediction function, `content_model` !!

    You must however change its contents (i.e. add your own content-based
    filtering algorithm), as well as altering/adding any other functions
    as part of your improvement.

    ---------------------------------------------------------------------

    Description: Provided within this file is a baseline content-based
    filtering algorithm for rating predictions on Movie data.

"""

# Script dependencies
import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from rake_nltk import Rake
import streamlit as st
# Importing data
movies = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/train.csv')
imdb = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/imdb_data.csv')
movies.dropna(inplace=True)
imdb.dropna(inplace=True)
#movies = pd.merge(movies, imdb[['movieId','plot_keywords']], on='movieId')
#movies.dropna(inplace=True)

#st.write(movies)
def data_preprocessing(subset_size):
    """Prepare data for use within Content filtering algorithm.

    Parameters
    ----------
    subset_size : int
        Number of movies to use within the algorithm.

    Returns
    -------
    Pandas Dataframe
        Subset of movies selected for content-based filtering.

    """
    # Split genre data into individual words.
    movies['keyWords'] = movies['genres'].str.replace('|', ' ')
    # Subset of the data
    movies_subset = movies[:subset_size]
    return movies_subset

# !! DO NOT CHANGE THIS FUNCTION SIGNATURE !!
# You are, however, encouraged to change its content.
def content_model(movie_list,top_n=10):
    """Performs Content filtering based upon a list of movies supplied
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
    movies1 = pd.merge(movies, imdb, on='movieId')
    movies1.dropna(inplace=True)
    df = movies1[['title','genres','director','title_cast','plot_keywords']]
    #rename columns
    df.columns = ['Title', 'Genre', 'Director', 'Actors', 'Plot']
    # discarding the commas between the actors' full names and getting only the first three names
    df.loc[:,'Actors'] = df.loc[:,'Actors'].map(lambda x: x.split('|')[:3])
    # putting the genres in a list of words
    df.loc[:,'Genre'] = df.loc[:,'Genre'].map(lambda x: x.lower().split('|'))

    df.loc[:,'Director'] = df.loc[:,'Director'].map(lambda x: x.split(' '))
    # merging together first and last name for each actor and director, so it's considered as one word
    # and there is no mix up between people sharing a first name
    for index, row in df.iterrows():
        row['Actors'] = [x.lower().replace(' ','') for x in row['Actors']]
        row['Director'] = ''.join(row['Director']).lower()   

    # initializing the new column
    df.loc[:,'Key_words'] = ""

    for index, row in df.iterrows():
        plot = row['Plot']        

        # instantiating Rake, by default is uses english stopwords from NLTK
        # and discard all puntuation characters
        r = Rake()      

        # extracting the words by passing the text
        r.extract_keywords_from_text(plot)  

        # getting the dictionary whith key words and their scores
        key_words_dict_scores = r.get_word_degrees()

        # assigning the key words to the new column
        row['Key_words'] = list(key_words_dict_scores.keys())  


    # dropping the Plot column
    df.drop(columns = ['Plot'], inplace = True)

    df.set_index('Title', inplace = True)

    df.loc[:,'bag_of_words'] = ''
    columns = df.columns

    for index, row in df.iterrows():
         words = ''
         for col in columns:
             if col != 'Director':
                 words = words + ' '.join(row[col])+ ' '
             else:
                 words = words + row[col]+ ' '
         row['bag_of_words'] = words
    # #dt = df['Title']
    df.drop(columns = [col for col in df.columns if col!= 'bag_of_words'], inplace = True)
    #df['bag_of_words'] = row['bag_of_words']
    #df['bag_of_words'] = row['bag_of_words']
    #df['Title'] = df.index
    df.reset_index(inplace = True)
    #df.reset_index(drop=True)
    # Subset of the data
    movies_subset = df[:27000]
    df_t = data_preprocessing(27000)

    # Initializing the empty list of recommended movies
    recommended_movies = []
    data = data_preprocessing(27000) #movies_subset               #data_preprocessing(27000)
    # Instantiating and generating the count matrix
    count_vec = CountVectorizer()
    count_matrix = count_vec.fit_transform(data['keyWords'])
    indices = pd.Series(data['title'])
    cosine_sim = cosine_similarity(count_matrix, count_matrix)
    # Getting the index of the movie that matches the title
    idx_1 = indices[indices == movie_list[0]].index[0]
    idx_2 = indices[indices == movie_list[1]].index[0]
    idx_3 = indices[indices == movie_list[2]].index[0]

    # Creating a Series with the similarity scores in descending order
    rank_1 = cosine_sim[idx_1]
    rank_2 = cosine_sim[idx_2]
    rank_3 = cosine_sim[idx_3]

    # Calculating the scores
    score_series_1 = pd.Series(rank_1).sort_values(ascending = False)
    score_series_2 = pd.Series(rank_2).sort_values(ascending = False)
    score_series_3 = pd.Series(rank_3).sort_values(ascending = False)
    # Getting the indexes of the 10 most similar movies
    listings = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)

    # Store movie names
    recommended_movies = []
    # Appending the names of movies
    top_50_indexes = list(listings.iloc[1:50].index)
    # Removing chosen movies
    top_indexes = np.setdiff1d(top_50_indexes,[idx_1,idx_2,idx_3])
    for i in top_indexes[:top_n]:
        recommended_movies.append(list(movies['title'])[i])
    return recommended_movies
