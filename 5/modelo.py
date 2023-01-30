import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler



dados = pd.read_csv('dados/movies.csv')[['movieId', 'genres']]
dados = dados.set_index('movieId')
dados_dummies = dados['genres'].str.get_dummies()
dados_dummies = dados_dummies.drop(columns='(no genres listed)')

dados_tags = pd.read_csv('dados/tags.csv')
dados_tags = dados_tags.drop(columns=['userId','timestamp'])
dados_tags = pd.DataFrame({'movieId':dados_tags['movieId'].unique(),
                      'tags':[dados_tags['tag'][dados_tags['movieId']==i].str.cat(sep="|")\
                              for i in dados_tags['movieId'].unique()]})
dados_tags = dados_tags.set_index('movieId')

dados_tags_dummies = dados_tags['tags'].str.get_dummies()
dados_tags_dummies = dados_tags_dummies.rename(columns={n:'tag_'+n for n in dados_tags_dummies.columns})
dados_dummies.join(dados_tags_dummies)

dados_ratings = pd.read_csv('dados/ratings.csv')
dados_ratings = dados_ratings.drop(columns=['userId','timestamp'])
dados_ratings = dados_ratings.groupby("movieId").mean()
scaler = StandardScaler()
scaler.fit(dados_ratings)
dados_ratings['rating'] = scaler.transform(dados_ratings)

dados_machine_learning = dados_dummies.join(dados_ratings)
dados_machine_learning['rating'][dados_machine_learning['rating'].isnull()] = 0


modelo_pca = PCA(n_components=3)
embedding_pca = modelo_pca.fit_transform(dados_machine_learning)
projection = pd.DataFrame(columns=['x', 'y', 'z'], data=embedding_pca)

modelo_kmeans_pca = KMeans(n_clusters=20)
modelo_kmeans_pca.fit(projection)
projection['cluster_pca'] = modelo_kmeans_pca.predict(projection)
projection = projection.join(pd.read_csv('dados/movies.csv'))
projection = projection.set_index('movieId')


def recomendador(nome_filme):
    cluster = list(projection[projection['title'] == nome_filme]['cluster_pca'])[0]
    filmes_recomendados = projection[projection['cluster_pca'] == cluster][['x', 'y','z', 'title', 'genres']]
    x = list(projection[projection['title'] == nome_filme]['x'])[0]
    y = list(projection[projection['title'] == nome_filme]['y'])[0]
    z = list(projection[projection['title'] == nome_filme]['z'])[0]
    distancias = euclidean_distances(filmes_recomendados[['x', 'y', 'z']], [[x, y, z]])
    filmes_recomendados['distancias'] = distancias
    return filmes_recomendados.sort_values('distancias').head(20)