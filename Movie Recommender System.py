#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')


# In[2]:


movies.head()


# In[3]:


movies.head(1)


# In[4]:


credits.head(1)


# In[5]:


movies = movies.merge(credits,on='title')


# In[6]:


movies.head(1)


# In[7]:


movies.info()


# In[8]:


# genres
# id
# keywords
# title
# overview
# cast
# crew

movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[9]:


movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]


# In[10]:


movies.head()


# In[11]:


movies.isnull().sum()


# In[12]:


movies.dropna()


# In[13]:


movies.dropna(inplace=True)


# In[14]:


movies.duplicated().sum()


# In[15]:


movies.iloc[0].genres


# In[16]:


# '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]'

# ['Action','Adventure','Fantasy','SciFi']


# In[17]:


def convert(obj):
    L = []
    for i in obj:
        L.append(i['name'])
    return L


# In[18]:


import ast
ast.literal_eval('[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}, {"id": 14, "name": "Fantasy"}, {"id": 878, "name": "Science Fiction"}]')


# In[19]:


def convert(obj):
    L = []
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L


# In[20]:


movies['genres'].apply(convert)


# In[21]:


movies['genres'] = movies['genres'].apply(convert)


# In[22]:


movies.head()


# In[23]:


movies['keywords'].apply(convert)


# In[24]:


movies['keywords'] = movies['keywords'].apply(convert)


# In[25]:


movies.head()


# In[26]:


def convert3(obj):
    L = []
    counter = 0
    for i in ast.literal_eval(obj):
        if counter !=3:
             L.append(i['name'])
             counter+=1
        else:
            break
    return L


# In[27]:


movies['cast'].apply(convert3)


# In[28]:


movies['cast'] = movies['cast'].apply(convert3)


# In[29]:


movies.head()


# In[30]:


movies['crew'][0]


# In[31]:


def fetch_director(obj):
    L = []
    for i in ast.literal_eval(obj):
        if i['job'] == 'Director':
            L.append(i['name'])
            break
    return L


# In[32]:


movies['crew'].apply(fetch_director)


# In[33]:


movies['crew'] = movies['crew'].apply(fetch_director)


# In[34]:


movies.head()


# In[35]:


movies['overview'][0]


# In[36]:


movies['overview'].apply(lambda x:x.split())


# In[37]:


movies['overview'] = movies['overview'].apply(lambda x:x.split())


# In[38]:


movies.head()


# In[39]:


movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[40]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])


# In[41]:


movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])


# In[42]:


movies.head()


# In[43]:


movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']


# In[44]:


movies.head()


# In[45]:


new_df = movies[['movie_id','title','tags']]


# In[46]:


new_df


# In[47]:


new_df['tags'].apply(lambda x:" ".join(x))


# In[48]:


new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))


# In[49]:


new_df.head()


# In[50]:


new_df['tags'][0]


# In[51]:


new_df['tags'].apply(lambda x:x.lower())


# In[52]:


new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())


# In[53]:


new_df.head()


# In[54]:


import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
    y = []

    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)   
        


# In[55]:


new_df['tags'].apply(stem)


# In[56]:


new_df['tags'] = new_df['tags'].apply(stem)


# In[57]:


from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')


# In[58]:


cv.fit_transform(new_df['tags']).toarray()


# In[59]:


cv.fit_transform(new_df['tags']).toarray().shape


# In[60]:


vectors = cv.fit_transform(new_df['tags']).toarray()


# In[61]:


vectors


# In[62]:


vectors[0]


# In[63]:


cv.get_feature_names_out()


# In[64]:


len(cv.get_feature_names_out())


# In[65]:


ps.stem('loved')


# In[66]:


stem('In the 22nd century, a paraplegic Marine is dispatched to the moon Pandora on a unique mission, but becomes torn between following orders and protecting an alien civilization. Action Adventure Fantasy ScienceFiction cultureclash future spacewar spacecolony society spacetravel futuristic romance space alien tribe alienplanet cgi marine soldier battle loveaffair antiwar powerrelations mindandsoul 3d SamWorthington ZoeSaldana SigourneyWeaver JamesCameron')


# In[67]:


from sklearn.metrics.pairwise import cosine_similarity


# In[68]:


cosine_similarity(vectors)


# In[69]:


cosine_similarity(vectors).shape


# In[70]:


similarity = cosine_similarity(vectors)


# In[71]:


similarity


# In[72]:


def recommend(movie):
    movie_index = new_df[new_df['title'] == movie].index[0]
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)),reverse=True,key = lambda x:x[1])[1:6]

    for i in movies_list:
        print(new_df.iloc[i[0]].title)


# In[156]:


recommend('Batman Begins')


# In[74]:


new_df.iloc[1216]


# In[75]:


new_df.iloc[1216].title


# In[76]:


new_df[new_df['title'] == 'Batman Begins'].index[0]


# In[77]:


sorted(list(enumerate(similarity[0])),reverse=True,key = lambda x:x[1])[1:6]


# In[161]:


get_ipython().system('pip install pickle')


# In[163]:


import pickle

pickle.dump(new_df,open('movies.pkl', 'wb'))


# In[167]:


new_df['title'].values


# In[169]:


new_df.to_dict()


# In[171]:


pickle.dump(new_df.to_dict(),open('movie_dict.pkl', 'wb'))


# In[173]:


pickle.dump(similarity,open('similarity.pkl', 'wb'))


# In[ ]:




