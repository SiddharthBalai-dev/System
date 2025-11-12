import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel, pairwise_distances
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Data
movies = pd.DataFrame({
 'title':['The Shawshank Redemption','The Godfather','The Dark Knight','Pulp Fiction','The Lord of the Rings: The Return of the King'],
 'genres':['Drama','Crime, Drama','Action, Crime, Drama','Crime, Drama','Action, Adventure, Fantasy'],
 'desc':['Two imprisoned men bond.','Patriarch transfers crime dynasty.','Joker wreaks havoc.','Mob hitmen tales.','Gandalf vs Sauron.']
})
movies['content']=movies['genres']+' '+movies['desc']

ratings=pd.DataFrame({
 'user_id':[1,1,1,2,2,3,3],
 'title':['The Shawshank Redemption','The Godfather','The Dark Knight','The Dark Knight','Pulp Fiction','The Shawshank Redemption','Pulp Fiction'],
 'rating':[5,4,5,4,5,5,4]
})

# Models
tfidf = TfidfVectorizer(stop_words='english')
cos_sim = linear_kernel(tfidf.fit_transform(movies['content']))
mat = csr_matrix(ratings.pivot(index='user_id', columns='title', values='rating').fillna(0).values)
latent = TruncatedSVD(2).fit_transform(mat)

# Functions
def content_rec(title):
    idx = movies.index[movies['title']==title][0]
    sim = sorted(list(enumerate(cos_sim[idx])), key=lambda x:x[1], reverse=True)[1:4]
    return [movies['title'][i[0]] for i in sim]

def collab_rec(uid):
    i = ratings['user_id'].unique().tolist().index(uid)
    sim = pairwise_distances(latent[i].reshape(1,-1), latent, metric='cosine')[0].argsort()[:3]
    rec = []
    [rec.extend(ratings[ratings['user_id']==ratings['user_id'].unique()[j]]['title']) for j in sim]
    return list(set(rec))

def hybrid(uid, title):
    return list(set(content_rec(title) + collab_rec(uid)))

# Output
print("Hybrid Recommendations:", hybrid(1, 'The Godfather'))
