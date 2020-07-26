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
from recommenders.Hybrid_Recommender_System import recommendation
from recommenders.Exploratory_data_analysis import count_word
from recommenders.Exploratory_data_analysis import random_color_func
import time
# Data Loading
title_list = load_movie_titles('resources/data/movies.csv')
rating_m = pd.read_csv('resources/data/ratings.csv')
imdb = pd.read_csv('~/unsupervised_data/unsupervised_movie_data/imdb_data.csv')
# App declaration
def main():

    # DO NOT REMOVE the 'Recommender System' option below, however,
    # you are welcome to add more options to enrich your app.
    page_options = ["Recommender System","Solution Overview","Exploratory Data Analysis","Search for a Movie","About Machine Learning App","Instruction of use","Hybrid Recommender System"]

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
        st.markdown('')
    # You may want to add more sections here for aspects such as an EDA,
    # or to provide your business pitch.
    #Search for a Movie page
    if page_selection  =="Search for a Movie":
        st.title("Search for Movies")
        st.markdown('Please Refer to the About Machine Learning Page to learn more about the techniques used to recommend movies. If you decide not to use the recommender systems you can use this page to filter movies based on the rating of the movie , the year in which the movie was released and the genre of the movies. After you change the filter you will be left with movies that are specific to that filter used. Then when you scroll down you will see the movie name and the link to a youtube trailer of that movie. When you click the link ,you will see a page on youtube for that specific movie and you can watch the trailer and see if you like it. This is an alternative method to you if you are not satisfied with the recommender engine . Enjoy! ')
        # Movies
        df = pd.read_csv('resources/data/movies.csv')
        #min_year = int(df['Year'].min())
        #max_year = int(df['Year'].max())
        #countries = df['Country Name'].unique()
        #'## By country'
        #country = st.selectbox('Country', countries)
        #df[df['Country Name'] == country]
        #movie_title = df['title'].unique()
        #title = st.selectbox('title', movie_title)
        ##################################################################################
        #movies_df[‘year’] = movies_df.title.str.extract(‘(\(\d\d\d\d\))’,expand=False)
        #Removing the parentheses
        #movies_df[‘year’] = movies_df.year.str.extract(‘(\d\d\d\d)’,expand=False)
        #Removing the years from the ‘title’ column
        #movies_df[‘title’] = movies_df.title.str.replace(‘(\(\d\d\d\d\))’, ‘’)
        #Applying the strip function to get rid of any ending whitespace characters that may have appeared
        #movies_df[‘title’] = movies_df[‘title’].apply(lambda x: x.strip())
        #########################################################################################
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

        movie_data = pd.merge(rating_m, df, on='movieId')
        #movie_data['timestamp'] = movie_data['timestamp'].apply(lambda x: time.strftime('%Y', time.localtime(x)))
        movie_data['year'] = movie_data.title.str.extract('(\(\d\d\d\d\))',expand=False)
        #Removing the parentheses
        movie_data['year'] = movie_data.year.str.extract('(\d\d\d\d)',expand=False)
        #movie_year = st.sidebar.slider("Pick the Year",1995,2019)
        movie_data.genres = movie_data.genres.str.split('|')
        #movie_title = movie_data['rating'].unique()
        movie_rating = st.sidebar.number_input("Pick a rating ",0.5,5.0, step=0.5)

        movie_data = explode(movie_data, ['genres'])
        movie_title = movie_data['genres'].unique()
        title = st.selectbox('Genre', movie_title)
        movie_data['year'].dropna(inplace = True)
        movie_data = movie_data.drop(['movieId','timestamp','userId'], axis = 1)
        year_of_movie_release = movie_data['year'].sort_values(ascending=False).unique()
        release_year = st.selectbox('Year', year_of_movie_release)

        movie = movie_data[(movie_data.rating == movie_rating)&(movie_data.genres == title)&(movie_data.year == release_year)]
        if len(movie) !=0:
            st.write(movie)
        if len(movie) ==0:
            st.write('We have no movies for that rating!')        
        def youtube_link(title):
    
            """This function takes in the title of a movie and returns a Search query link to youtube
    
            INPUT: ('The Lttle Mermaid')
            -----------
    
            OUTPUT: https://www.youtube.com/results?search_query=The+little+Mermaid&page=1
            ----------
            """
            title = title.replace(' ','+')
            base = "https://www.youtube.com/results?search_query="
            q = title
            page = "&page=1"
            URL = base + q + page
            return URL            
        if len(movie) !=0:           
            for _, row in movie.iterrows():
                st.write(row['title'])
                st.write(youtube_link(title = row['title']))
    # Building out the EDA page
    if page_selection == "Exploratory Data Analysis":
        st.title("Insights on how people rate movies")       
        if st.checkbox('Show Rating graph'):
            rating_m.groupby('rating')['userId'].count().plot(kind = 'bar', color = 'g',figsize = (8,7))
            plt.xticks(rotation=85, fontsize = 14)
            plt.yticks(fontsize = 14)
            plt.xlabel('Ratings (scale: 0.5 - 5.0)', fontsize=16)
            plt.ylabel('No. of Ratings', fontsize=16)
            plt.title('Distribution of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot()
            st.markdown("This is a bar graph showing the rating of movie by people who have watched them.")
            st.markdown("The number of ratings is the total number of rating for each scale from 0.5 upto 5.0 rated by people who watched the movies.")
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
            labels = 'Low Ratings', 'Medium Ratings', 'High Ratings'
            sizes = [low_ratings, medium_ratings,  high_ratings]
            explode = (0, 0, 0.1)  # Only "explore" the 3rd slice (i.e. 'Anti')

            # Create pie chart with the above labels and calculated class proportions as inputs
            fig1, ax1 = plt.subplots()
            ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
                    shadow=True, startangle=270)#,textprops={'rotation': 65}
            ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Categorised Proportions of User Ratings ',bbox={'facecolor':'k', 'pad':5},color='w',fontsize = 18)
            st.pyplot()
            st.markdown("This is a pie chart showing the rating of movies by people who have watched them.")
            st.markdown("Low Ratings (scale: 0.5 - 2.0)")
            st.markdown("Medium Ratings (scale: 2.5 - 3.5)")
            st.markdown("High Ratings (scale: 4.0 - 5.0)")

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
            wordcloud = WordCloud(width=1000, height=800).generate(directors)

            #ploting the word cloud
            plt.figure(figsize=(16,12))
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot() 
            st.markdown("This is a wordcloud of the directors of movies in this Application.")
            st.markdown("This wordcloud shows the most popular directors on the movies.")
        if st.checkbox('Show WordCloud of Actors/Actresses'):
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
            wordcloud = WordCloud(width=1000, height=800).generate(title_cast)

            #ploting the word cloud
            plt.figure(figsize=(16,12))
            plt.imshow(wordcloud)
            plt.axis('off')
            st.pyplot()  
            st.markdown("This is a wordcloud for Actors/Actresses on the movies on this Application.")
            st.markdown("This wordcloud shows the most popular Actors/Actresses on the movies.")
        if st.checkbox("Show wordcloud of different genres"):    
            movies = pd.read_csv('resources/data/movies.csv')
            #here we  make census of the genres:
            genre_labels = set()
            for s in movies['genres'].str.split('|').values:
                genre_labels = genre_labels.union(set(s))  

            #counting how many times each of genres occur:
            keyword_occurences, dum = count_word(movies, 'genres', genre_labels)
            #Finally, the result is shown as a wordcloud:
            words = dict()
            trunc_occurences = keyword_occurences[0:50]
            for s in trunc_occurences:
                words[s[0]] = s[1]
            tone = 100 # define the color of the words
            f, ax = plt.subplots(figsize=(14, 6))
            wordcloud = WordCloud(width=1000,height=800, background_color='white', 
                                max_words=1628,relative_scaling=0.7,
                                color_func = random_color_func,
                                normalize_plurals=False)
            wordcloud.generate_from_frequencies(words)
            plt.figure(figsize=(16,12))
            plt.imshow(wordcloud, interpolation="bilinear")
            plt.axis('off')
            st.pyplot()
            st.markdown("This is a wordcloud for all the different genres in this Application.")

    if page_selection == "Hybrid Recommender System":
        st.image('resources/imgs/Image_header.png',use_column_width=True)
        #st.write('### Enter Your User Id')
        #user_Id = rating_m['userId'][:1000]
        #user_Id = rating_m['userId'].unique()
        #user_id1 = st.selectbox('Fisrt Option',user_Id[:1000])
        title_list1 = load_movie_titles('recommenders/hybrid_movies.csv')
        #title_list1 = title_list1.unique()
        # User-based preferences
        st.write('### Enter Your Three Favorite Movies')
        movie_1 = st.selectbox('Fisrt Option',title_list1[200:500])
        movie_2 = st.selectbox('Second Option',title_list1[700:1000])
        movie_3 = st.selectbox('Third Option',title_list1[2000:2300])
        fav_movies = [movie_1,movie_2,movie_3]
        #st.write(movie_1)
        #fav_movies = [movie_1]
        if st.button("Recommend"):
            #try:
                with st.spinner('Crunching the numbers...'):
                    top_recommendations = recommendation(fav_movies, top_n = 10)
                    #st.write(top_recommendations)
                st.title("We think you'll like:")
                for i,j in enumerate(top_recommendations):
                    #st.write(str(i+1),j)
                    st.subheader(str(i+1)+'. '+j)

    if page_selection == "Instruction of use":
        st.title("Instructions")
        st.markdown('When the application opens the first page you will see is the Recommender System Page. Here you will see two algorithms you can choose from.')
        st.image('recommenders/images/page1.png', width = 600)
        st.markdown('Then you will have three options to choose from. On the select box you will choose your three favourite movie and then press the Recommend button.')
        st.image(['recommenders/images/page2.png','recommenders/images/page3.png'],width=600)
        st.markdown('After pressing the Recommend button then the recommended movies will be shown to you.')
        st.image('recommenders/images/page4.png',width = 600)
        st.markdown('Then you can choose the next algorithm to do the same task.')
        st.image('recommenders/images/page16.png',width = 600)
        st.markdown('On the left you will see a side bar that has all of the pages on this App.')
        st.image('recommenders/images/page5.png',width = 600)
        st.markdown('We have another page where we show you Actors , directors and how other people that watched the movies rate them.')
        st.image('recommenders/images/page18.png',width = 600 )
        st.markdown('This word cloud show the most popular Actors/Actresses on this movie App.')
        st.image('recommenders/images/page7.png',width = 600)
        st.markdown('This word cloud shows the different genres you can find on this App.')
        st.image('recommenders/images/page8.png',width = 600)
        st.markdown('The Search for a Movie page is an alternative to search for movies using filters instead of using Recommender Systems . The filters you can use are the release year of a movie, the rating of a movie and the genre of a movie and this will allow you to play around and find a specific movie using filters.')
        st.image(['recommenders/images/page10.png','recommenders/images/page11.png'],width = 600)
        st.image(['recommenders/images/page12.png','recommenders/images/page13.png'],width = 600)
        st.markdown('Whe you scroll down this page you will see that as you change the movie name and a link . The link is a youtube movie trailer for that movie.')
        st.image('recommenders/images/page14.png',width = 600)
        st.markdown('Then when you click the link it will take you to the youtube page for the trailer of that movie.')
        st.image('recommenders/images/page15.png',width = 600)

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
        st.markdown('<p>Content-based filtering approaches utilize a series of discrete, pre-tagged characteristics of an item in order to recommend additional items with similar properties. Current recommender systems typically combine one or more approaches into a hybrid system. </p>', unsafe_allow_html=True)
        st.subheader('Hybrid filtering')
        st.markdown(' Hybrid filtering technique is a combination of multiple recommendation techniques like, merging collaborative filtering with content-based filtering or vice-versa.')
        st.markdown(' Most recommender systems now use a hybrid approach, combining collaborative filtering, content-based filtering, and other approaches . There is no reason why several different techniques of the same type could not be hybridized. Hybrid approaches can be implemented in several ways: by making content-based and collaborative-based predictions separately and then combining them; by adding content-based capabilities to a collaborative-based approach (and vice versa); or by unifying the approaches into one model .')
        st.markdown(' For more information about building data Apps Please go to :<a href="https://www.streamlit.io/">streamlit site</a></p>', unsafe_allow_html=True)	
        st.markdown('<p> </p>', unsafe_allow_html=True)	

if __name__ == '__main__':
    main()
