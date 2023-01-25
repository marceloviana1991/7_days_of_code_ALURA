import pandas as pd
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

dados = pd.read_csv('movies.csv')[['movieId', 'genres']]
dados = dados.set_index('movieId')
dados = dados['genres'].str.get_dummies()
dados = dados.drop(columns='(no genres listed)')

modelo_pca = PCA(n_components=2)
embedding_pca = modelo_pca.fit_transform(dados)
projection = pd.DataFrame(columns=['x', 'y'], data=embedding_pca)

modelo_kmeans_pca = KMeans(n_clusters=20)
modelo_kmeans_pca.fit(projection)
projection['cluster_pca'] = modelo_kmeans_pca.predict(projection)
projection = projection.join(pd.read_csv('movies.csv'))
projection = projection.set_index('movieId')


def recomendador(nome_filme):
    cluster = list(projection[projection['title'] == nome_filme]['cluster_pca'])[0]
    filmes_recomendados = projection[projection['cluster_pca'] == cluster][['x', 'y', 'title', 'genres']]
    x = list(projection[projection['title'] == nome_filme]['x'])[0]
    y = list(projection[projection['title'] == nome_filme]['y'])[0]
    distancias = euclidean_distances(filmes_recomendados[['x', 'y']], [[x, y]])
    filmes_recomendados['distancias'] = distancias
    return filmes_recomendados.sort_values('distancias').head(100)