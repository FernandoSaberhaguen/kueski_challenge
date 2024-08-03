import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# Supongamos que el DataFrame 'df' ya está cargado en memoria
df_unique = df.drop_duplicates(subset='clean_title')

# Crear una descripción combinada de características de las películas
df_unique['combined_features'] = df_unique.apply(lambda row: ' '.join(
    [str(row['Comedy']), str(row['Horror']), str(row['Sci-Fi']), str(row['Action']), str(row['Romance']), 
     str(row['Animation']), str(row['Drama']), row['clean_title'], row['tag_x']]), axis=1)

# Vectorizar las características combinadas
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_unique = tfidf_vectorizer.fit_transform(df_unique['combined_features'])

# Entrenar el modelo de NearestNeighbors con los mejores parámetros
nn_model_unique = NearestNeighbors(metric='cosine', algorithm='auto', n_neighbors=15)
nn_model_unique.fit(tfidf_matrix_unique)

# Función para obtener recomendaciones basadas en contenido utilizando NearestNeighbors con datos únicos
def get_content_based_recommendations_nn_unique(title, nn_model=nn_model_unique, tfidf_matrix=tfidf_matrix_unique, n_recommendations=10):
    if title not in df_unique['clean_title'].values:
        return "La película no se encuentra en el conjunto de datos."
    idx = df_unique[df_unique['clean_title'] == title].index[0]
    distances, indices = nn_model.kneighbors(tfidf_matrix[idx], n_neighbors=n_recommendations+1)
    indices = indices.flatten()[1:]
    return df_unique['clean_title'].iloc[indices]

# Crear una matriz de usuario-película
df_dedup = df.drop_duplicates(subset=['userId', 'clean_title'])
user_movie_ratings = df_dedup.pivot(index='userId', columns='clean_title', values='rating').fillna(0)
user_movie_ratings_sparse = csr_matrix(user_movie_ratings.values)

# Definir y entrenar el modelo de SVD con los mejores parámetros
svd_model = TruncatedSVD(n_components=5, algorithm='arpack', random_state=42)
user_factors = svd_model.fit_transform(user_movie_ratings_sparse)
user_similarities_svd = cosine_similarity(user_factors)

# Función para obtener recomendaciones basadas en colaboración utilizando SVD
def get_user_based_recommendations_svd(user_id, n_recommendations=10):
    if user_id not in user_movie_ratings.index:
        return "El usuario no se encuentra en el conjunto de datos."
    user_idx = user_movie_ratings.index.get_loc(user_id)
    sim_scores = user_similarities_svd[user_idx]
    sim_scores_idx = sim_scores.argsort()[::-1]
    top_user_indices = sim_scores_idx[1:n_recommendations+1]
    similar_users_ratings = user_movie_ratings.iloc[top_user_indices]
    recommended_movies = similar_users_ratings.mean(axis=0).sort_values(ascending=False).head(n_recommendations)
    return recommended_movies.index

def get_real_time_recommendations(title=None, user_id=None, n_recommendations=10):
    recommendations = {}
    if title:
        content_recs = get_content_based_recommendations_nn_unique(title, n_recommendations=n_recommendations)
        recommendations['content_based'] = content_recs
    if user_id:
        collab_recs = get_user_based_recommendations_svd(user_id, n_recommendations=n_recommendations)
        recommendations['collaborative_based'] = collab_recs
    return recommendations
