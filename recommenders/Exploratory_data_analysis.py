

import pandas as pd

# To create plots
import matplotlib.pyplot as plt # data visualization library
import seaborn as sns
sns.set_style('whitegrid')
from wordcloud import WordCloud, STOPWORDS #used to generate world cloud



#define a function that counts the number of times each genre appear:
def count_word(df, ref_col, lister):
    keyword_count = dict()
    for s in lister: keyword_count[s] = 0
    for lister_keywords in df[ref_col].str.split('|'):
        if type(lister_keywords) == float and pd.isnull(lister_keywords): continue
        for s in lister_keywords: 
            if pd.notnull(s): keyword_count[s] += 1
    # convert the dictionary in a list to sort the keywords  by frequency
    keyword_occurences = []
    for k,v in keyword_count.items():
        keyword_occurences.append([k,v])
    keyword_occurences.sort(key = lambda x:x[1], reverse = True)
    return keyword_occurences, keyword_count


    
# Function that control the color of the words
def random_color_func(word=None, font_size=None, position=None,
                      orientation=None, font_path=None, random_state=None):
    h = int(360.0 * tone / 255.0)
    s = int(100.0 * 255.0 / 255.0)
    l = int(100.0 * float(random_state.randint(70, 120)) / 255.0)
    return "hsl({}, {}%, {}%)".format(h, s, l)