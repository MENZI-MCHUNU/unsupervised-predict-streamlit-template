"""

    Streamlit webserver-based Recommender Engine.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within the root of this repository for guidance on how to use
    this script correctly.

    NB: !! Do not remove/modify the code delimited by dashes !!

    This application is intended to be partly marked in an automated manner.
    Altering delimited code may result in a mark of 0.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend certain aspects of this script
    and its dependencies as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st

# Data handling dependencies
import pandas as pd
import numpy as np
# To create plots
import matplotlib.pyplot as plt # data visualization library
import seaborn as sns
sns.set_style('whitegrid')
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud
# Custom Libraries
from utils.data_loader import load_movie_titles
from recommenders.collaborative_based import collab_model
from recommenders.content_based import content_model

# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
rating_m = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/imdb_data.csv')
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Exploratory Data Analysis","About Machine Learning App","Instruction of use"]

    # -------------------------------------------------------------------
    # ----------- !! THIS CODE MUST NOT BE ALTERED !! -------------------
    # -------------------------------------------------------------------
    page_selection = st.sidebar.selectbox("Choose Option", page_options)
    if page_selection == "Recommender System":
        # Header contents
        st.write('# Movie Recommender Engine')
        st.write('### EXPLORE Data Science Academy Unsupervised Predict')
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        # Recommender System algorithm selection
        sys = st.radio("Select an algorithm",
                       ('Content Based Filtering',
                        'Collaborative Based Filtering'))

        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list[14930:15200])
        movie_2 = st.selectbox('Second Option',title_list[25055:25255])
        movie_3 = st.selectbox('Third Option',title_list[21100:21200])
        fav_movies = [movie_1,movie_2,movie_3]

        # Perform top-10 movie recommendation generation
        if sys == 'Content Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = content_model(movie_list=fav_movies,
                                                            top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


        if sys == 'Collaborative Based Filtering':
            if st.button("Recommend"):
                try:
                    with st.spinner('Crunching the numbers...'):
                        top_recommendations = collab_model(movie_list=fav_movies,
                                                           top_n=10)
                    st.title("We think you'll like:")
                    for i,j in enumerate(top_recommendations):
                        st.subheader(str(i+1)+'. '+j)
                except:
                    st.error("Oops! Looks like this algorithm does't work.\
                              We'll need to fix it!")


    # -------------------------------------------------------------------

    # ------------- SAFE FOR ALTERING/EXTENSION -------------------
    if page_selection == "Solution Overview":
        st.title("Solution Overview")
        st.write("Describe your winning approach on this page")

    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    # Building out the EDA page
    if page_selection == "Exploratory Data Analysis":
        st.title("Insights on how people rate movies")
        st.subheader("Rating Data")
        if st.checkbox('Show Rating data'):
            st.write(rating_m[['userId','movieId','rating']])
        if st.checkbox('Show Rating bar graph'):    
            num_users = len(rating_m.userId.unique())
            num_items = len(rating_m.movieId.unique())
            st.markdown('There are {} unique users and {} unique movies in this data set'.format(num_users, num_items))
            # get count
            df_ratings_cnt_tmp = pd.DataFrame(rating_m[['userId','movieId','rating']].groupby('rating').size(), columns=['count'])           
            # there are a lot more counts in rating of zero
            total_cnt = num_users * num_items
            rating_zero_cnt = total_cnt - rating_m[['userId','movieId','rating']].shape[0]
            # append counts of zero rating to df_ratings_cnt
            df_ratings_cnt = df_ratings_cnt_tmp.append(
                    pd.DataFrame({'count': rating_zero_cnt}, index=[0.0]),
                    verify_integrity=True,
            ).sort_index()            
            # add log count
            df_ratings_cnt['log_count'] = np.log(df_ratings_cnt['count'])
 
            ax = df_ratings_cnt[['count']].reset_index().rename(columns={'index': 'rating score'}).plot(
                        x='rating score',
                        y='count',
                        kind='bar',
                        figsize=(12, 8),
                        title='Count for Each Rating Score (in Log Scale)',
                        logy=True,
                        fontsize=12,
            )
            ax.set_xlabel("movie rating score")
            ax.set_ylabel("number of ratings") 
            st.pyplot()  
        if st.checkbox('Show Rating graph'):
            rating_m.groupby('rating')['userId'].count().plot(kind = 'bar', color = 'g',figsize = (8,7))
            plt.xticks(rotation=85, fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.xlabel('Ratings (scale: 0.5 - 5.0)', fontsize=16)
            plt.ylabel('No. of Ratings', fontsize=16)
            plt.title('Distribution of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot()
        if st.checkbox('Show Pie chart for ratings'):
            # Calculate and categorise ratings proportions
            a = len(rating_m.loc[rating_m['rating']== 0.5]) / len(rating_m)
            b = len(rating_m.loc[rating_m['rating']==1.0]) / len(rating_m)
            c = len(rating_m.loc[rating_m['rating']==1.5]) / len(rating_m)
            d = len(rating_m.loc[rating_m['rating']==2.0]) / len(rating_m)
            low_ratings= a+b+c+d
            e = len(rating_m.loc[rating_m['rating']==2.5]) / len(rating_m)
            f = len(rating_m.loc[rating_m['rating']== 3.0]) / len(rating_m)
            g = len(rating_m.loc[rating_m['rating']==3.5]) / len(rating_m)
            medium_ratings= e+f+g
            h = len(rating_m.loc[rating_m['rating']==4.0]) / len(rating_m)
            i = len(rating_m.loc[rating_m['rating']==4.5]) / len(rating_m)
            j = len(rating_m.loc[rating_m['rating']==5.0]) / len(rating_m)
            high_ratings= h+i+j 
            # To view proportions of ratings categories, it is best practice to use pie charts
            # Where the slices will be ordered and plotted clockwise:
            labels = 'Low Ratings (scale: 0.5 - 2.0)', 'Medium Ratings (scale: 2.5 - 3.5)', 'High Ratings (scale: 4.0 - 5.0)'
            sizes = [low_ratings, medium_ratings,  high_ratings]
            explode = (0, 0, 0.1)  # Only "explore" the 3rd slice (i.e. 'Anti')

            # Create pie chart with the above labels and calculated class proportions as inputs
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=270)
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Categorised Proportions of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot()
                      
        if st.checkbox('Show WordCloud of directors'):   
            imdb["title_cast"] = imdb["title_cast"].astype('str')
            imdb["director"] = imdb["director"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('|',' '))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace(' ',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('|',' '))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace(' ',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('Seefullsummary',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('nan',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('nan',''))
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('nan',''))  

            directors = ' '.join([text for text in imdb["director"]])

            # Word cloud for the overall data checking out which words do people use more often
            wordcloud = WordCloud(width=900, height=600,random_state=21,max_font_size=110).generate(directors)

            #ploting the word cloud
            plt.figure(figsize=(16,8))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot() 

        if st.checkbox('Show WordCloud of Actors'):
            imdb["title_cast"] = imdb["title_cast"].astype('str')
            imdb["director"] = imdb["director"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].astype('str')
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('|',' '))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace(' ',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('|',' '))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace(' ',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('Seefullsummary',''))
            imdb["director"] = imdb["director"].apply(lambda x: x.replace('nan',''))
            imdb["title_cast"] = imdb["title_cast"].apply(lambda x: x.replace('nan',''))
            imdb["plot_keywords"] = imdb["plot_keywords"].apply(lambda x: x.replace('nan',''))   
                      
            title_cast= ' '.join([text for text in imdb["title_cast"]])

            # Word cloud for the overall data checking out which words do people use more often
            wordcloud = WordCloud(width=900, height=600,random_state=21,max_font_size=110).generate(title_cast)

            #ploting the word cloud
            plt.figure(figsize=(16,8))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot()  
                                                                                 
    # Building out the About Machine Learning App page
    if page_selection == "About Machine Learning App":
        st.title("Welcome to the Recommender System Machine Learning App")
        st.subheader('Machine Learning')
        st.markdown('<p>Machine learning (ML) is the study of computer algorithms that improve automatically through experience.It is seen as a subset of artificial intelligence. Machine learning algorithms build a mathematical model based on sample data, known as "training data", in order to make predictions or decisions without being explicitly programmed to do so.Machine learning algorithms are used in a wide variety of applications, such as email filtering and computer vision, where it is difficult or infeasible to develop conventional algorithms to perform the needed tasks. </p>', unsafe_allow_html=True)       
        st.subheader('Machine Learning Algorithms')
        st.markdown('<p>A machine learning (ML) algorithm is essentially a process or sets of procedures that helps a model adapt to the data given an objective. An ML algorithm normally specifies the way the data is transformed from input to output and how the model learns the appropriate mapping from input to output. </p>', unsafe_allow_html=True)
        st.subheader('Recommendation System')
        st.markdown('<p>A recommendation system  is a subclass of information filtering system that seeks to predict the "rating" or "preference" a user would give to an item. They are primarily used in commercial applications.Recommender systems are utilized in a variety of areas and are most commonly recognized as playlist generators for video and music services like Netflix, YouTube and Spotify, product recommenders for services such as Amazon, or content recommenders for social media platforms such as Facebook and Twitter. These systems can operate using a single input, like music, or multiple inputs within and across platforms like news, books, and search queries. Recommender systems usually make use of either or both collaborative filtering and content-based filtering.<p/>', unsafe_allow_html=True)
        st.subheader('Collaborative filtering')
        st.markdown('<p> Collaborative filtering approaches build a model from a user\'s past behavior (items previously purchased or selected and/or numerical ratings given to those items) as well as similar decisions made by other users. This model is then used to predict items (or ratings for items) that the user may have an interest in.<p/>', unsafe_allow_html=True)
        st.subheader('Content-based filtering')
        st.markdown('<p>Content-based filtering approaches utilize a series of discrete, pre-tagged characteristics of an item in order to recommend additional items with similar properties. Current recommender systems typically combine one or more approaches into a hybrid system.</p>', unsafe_allow_html=True)
        st.markdown('<p>For more information about building data Apps Please go to :<a href="https://www.streamlit.io/">streamlit site</a></p>', unsafe_allow_html=True)	
        st.markdown('<p> </p>', unsafe_allow_html=True)	

if __name__ == '__main__':
    main()
