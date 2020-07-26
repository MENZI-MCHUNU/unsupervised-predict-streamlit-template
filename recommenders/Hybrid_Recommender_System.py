"""


"""
import streamlit as st
#Importing the required libraries
import pandas as pd 
import numpy as np
import random
#from playsound import playsound
#import speech_recognition as sr
import re 
#import gtts
from surprise import Reader, Dataset, SVD
from sklearn.metrics.pairwise import cosine_similarity


#importing the dataset
movies = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/movies.csv', sep = ',',delimiter=',')
ratings = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/train.csv')
movies = movies[:20000]
#movies =  movies.reset_index(drop=True)
ratings = ratings.sample(frac=0.05)
ratings =  ratings.reset_index(drop=True)

def explode(df, lst_cols, fill_value='', preserve_index=False):
    import numpy as np
    # make sure `lst_cols` is list-alike
    if (lst_cols is not None
        and len(lst_cols) > 0
        and not isinstance(lst_cols, (list, tuple, np.ndarray, pd.Series))):
        lst_cols = [lst_cols]
    # all columns except `lst_cols`
    idx_cols = df.columns.difference(lst_cols)
    # calculate lengths of lists
    lens = df[lst_cols[0]].str.len()
    # preserve original index values    
    idx = np.repeat(df.index.values, lens)
    # create "exploded" DF
    res = (pd.DataFrame({
                col:np.repeat(df[col].values, lens)
                for col in idx_cols},
                index=idx)
             .assign(**{col:np.concatenate(df.loc[lens>0, col].values)
                            for col in lst_cols}))
    # append those rows that have empty lists
    if (lens == 0).any():
        # at least one list in cells is empty
        res = (res.append(df.loc[lens==0, idx_cols], sort=False)
                  .fillna(fill_value))
    # revert the original index order
    res = res.sort_index()
    # reset index if requested
    if not preserve_index:        
        res = res.reset_index(drop=True)
    return res

 
movies1=movies.copy()
movies1.genres = movies1.genres.str.split('|')
movies = explode(movies1, ['genres'])  


movies['year'] = movies.title.str.extract('(\(\d\d\d\d\))',expand=False)
#Removing the parentheses
movies['year'] = movies.year.str.extract('(\d\d\d\d)',expand=False)
#Removing the years from the title column
#movies['title'] = movies.title.str.replace('(\(\d\d\d\d\))', '')

#Applying the strip function to get rid of any ending whitespace characters that may have appeared
#movies['title'] = movies['title'].apply(lambda x: x.strip())

movies.to_csv('hybrid_movies.csv')
'''Applying the Cotent_Based Filtering'''
 #Applying Feature extraction 
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf=TfidfVectorizer(stop_words='english')
#matrix after applying the tfidf
matrix=tfidf.fit_transform(movies['genres'])


#Compute the cosine similarity of every genre
from sklearn.metrics.pairwise import cosine_similarity

cosine_sim=cosine_similarity(matrix,matrix)

'''Applying the Collaborative Filtering'''
#Intialising the Reader which is used to parse the file containing the ratings 
reader=Reader()

#Making the dataset containing the column as userid itemid ratings
#the order is very specific and we have to follow the same order
rating_dataset = Dataset.load_from_df(ratings[['userId','movieId','rating']],reader)

#Intialising the SVD model and specifying the number of latent features
#we can tune this parameters according to our requirement
svd=SVD(n_factors=25)

#making the dataset to train our model
training = rating_dataset.build_full_trainset()
#training our model
svd.fit(training)

#Making a new series which have two columns in it 
#Movie name and movie id 
movies_dataset = movies.reset_index()
titles = movies_dataset['title']
indices = pd.Series(movies_dataset.index, index=movies_dataset['title'])
indices1 = pd.Series(movies_dataset.index, index=movies_dataset['title'])
#Function to make recommendation to the user
def recommendation(movie_list,top_n):
    result=[]
    #Getting the id of the movie for which the user want recommendation
    st.write(movie_list[0])
    st.write(movie_list[1])
    st.write(movie_list[2])
    ind=indices[movie_list[0]].iloc[0]
    ind1=indices1[movie_list[1]].iloc[0]
    ind2=indices[movie_list[2]].iloc[0]
    #np.where(v == maximum)
    #st.write(ind)
    #Getting all the similar cosine score for that movie
    sim_scores=list(enumerate(cosine_sim[ind]))
    sim_scores1=list(enumerate(cosine_sim[ind1]))
    sim_scores2=list(enumerate(cosine_sim[ind2]))

    # Calculating the scores
    score_series_1 = pd.Series(sim_scores).sort_values(ascending = False)
    score_series_2 = pd.Series(sim_scores1).sort_values(ascending = False)
    score_series_3 = pd.Series(sim_scores2).sort_values(ascending = False)

    sim_scores_1 = score_series_1.append(score_series_2).append(score_series_3).sort_values(ascending = False)
    #st.write(sim_scores)
    #Sorting the list obtained
    sim_scores=sorted(sim_scores_1,key=lambda x:x[1],reverse=True)    
    #Getting all the id of the movies that are related to the movie Entered by the user
    movie_id=[i[0] for i in sim_scores]    
    print('The Movie You Should Watched Next Are --')
    print('ID ,   Name ,  Average Ratings , Year ')
    #st.write(movie_id)
    #Varible to print only top 10 movies
    count=0
    for id in range(0,len(movie_id)):
      #to ensure that the movie entered by the user is doesnot come in his/her recommendation
        if(ind != movie_id[id]):
            rating=ratings[ratings['movieId']==movie_id[id]]['rating']
            avg_ratings=round(np.mean(rating),2)
            #To print only thoese movies which have an average ratings that is more than 3.5
            if(avg_ratings >3.5):
                count+=1
                print(f'{movie_id[id]} , {titles[movie_id[id]]} ,{avg_ratings}')
                result.append(titles[movie_id[id]])#,avg_ratings])
            if(count >=top_n):
                    break
    
    print('Wait!! i am telling your recommendation')
    return result