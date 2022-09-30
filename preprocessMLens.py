import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow import keras
from keras import layers


def preprocessMovieLens1M(path_input, sparsity, dim_movie_contexts=18, top_users=1000000, numClust=100, n=50,
                          embeddingSize=50):
    # get the user data
    users = pd.read_cv(path_input + "/users.csv", header=0, sep=",")
    users.columns = ["user id", "gender", "age", "occupation", "zipcode"]
    # dropping zipcode, can keep and hot-encode it as well but potentially need to preprocess first and get 'regions'
    users = users.drop(["zipcode"], axis=1)

    # aggregate the age
    bins = [0, 19, 29, 39, 49, np.inf]
    names = ['<20', '20-29', '30-39', '40-49', '50+']
    users['agegroup'] = pd.cut(users['age'], bins, labels=names)
    users = users.drop(["age"], axis=1)

    # encode all information in binary fashion
    columnsToEncode = ["agegroup", "gender", "occupation"]
    myEncoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    myEncoder.fit(users[columnsToEncode])
    user_features = pd.concat([users.drop(columnsToEncode, 1),
                               pd.Dataframe(myEncoder.transform(users[columnsToEncode]),
                                            columns=myEncoder.get_feature_names(columnsToEncode))], axis=1).reindex()

    # get the movie dota
    movie_raw = pd.read_csv(path_input + "/movies.csv", header=0, sep=',')
    movie = pd.DataFrame(columns=["movie_id", "Action", "Adventure", "Animation", "Children's", "Comedy", "Crime",
                                  "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical", "Mystery",
                                  "Romance", "Sci-Fi", "Thriller", "War", "Western"])
    movie["movie_id"] = movie_raw["movield"]
    movie = movie.fillna(0)
    # binary encode the genres starting from a List of genres in the original dataset
    for i in range(len(movie_raw)):
        for j in range(dim_movie_contexts):
            if movie.columns[1 + j] in movie_raw['genres'][1]:
                movie[movie.columns[1 + j]][i] = 1
    movie_features = movie

    # this will be an input to my model, although this might be something that can be induced from my outputs t#
    dim_movie_contexts = movie.to_numpy().shape[1]

    # get the ratings data
    data = pd.read_csv(path_input + "/ratings.csv", sep=",", header=0,
                       names=["user id", "movie_id", "rating", "timestamp"])
    top_users = min(top_users, len(user_features))
    # Find total number of ratings by the top top_users users with the most ratings given out
    # data.groupby("user_id").count().sort_values( "rating", ascending = False).head(top_users)["rating"].sum()
    users_to_be_filtered = pd.DataFrame(
        data.groupby("user id").count().sort_values("rating", ascending=False).head(top_users).index)
    users_to_be_filtered["cluster"] = np.array([range(top_users)]).T.astype(int)

    # this is the (user-filtered) data we will use to decide on the top n movies
    user_filtered_data = data[data["user id"].isin(users_to_be_filtered["user id"])]
    # Obtain n top movies index
    top_movies_index = \
        user_filtered_data.groupby("movie_id").count().sort_values("rating", ascending=False).head(n).reset_index()[
            "movie id"]
    top_movies_features = movie_features[movie_features.movie_id.isin(top_movies_index)]
    top_movies_indices = top_movies_index.to_numpy()

    filtered_data_original = user_filtered_data[user_filtered_data["movie_id"].isin(top_movies_index)]
    df = filtered_data_original.copy()

    # Collaborative filtering run is here # 6599, 6601
    user_ids = users_to_be_filtered["user id"].unique().tolist()
    user2user_encoded = {x: i for i, x in enumerate(user_ids)}
    userencoded2user = {i: x for i, x in enumerate(user_ids)}
    movie_ids = df["movie id"].unique().tolist()
    movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
    movie_encoded2movie = {i: i for i, x in enumerate(movie_ids)}
    df["user"] = df["user id"].map(user2user_encoded)
    df["movie"] = df["movie _id"].map(movie2movie_encoded)

    num_users = len(user2user_encoded)
    num_movies = len(movie_encoded2movie)
    df["rating"] = df["rating"].values.astype(np.float32)
    # min and max ratings will be used in normalization
    min_rating = min(df["rating"])
    max_rating = max(df["rating"])

    df = df.sample(frac=1, random_state=42)
    x = df[["user", "movie"]].values
    # Normalize the targets between 0 and 1. Makes it easy to train
    y = df["rating"].apply(lambda x: (x - min_rating) / (max_rating - min_rating)).values
    # Assuming training on 90% of the data and validating on 10 %.
    train_indices = int(0.9 * df.shape[0])
    x_train, x_val, y_train, y_val = (
        x[:train_indices],
        x[train_indices:],
        y[:train_indices],
        y[train_indices:]
    )

    class RecommenderNet(keras.Model):
        def __init__(self, num_users_, num_movies_, embedding_size, **kwargs):
            super(RecommenderNet, self).__init__(**kwargs)
            self.num_users = num_users_
            self.num_movies = num_movies_
            self.embedding_size = embedding_size
            self.user_embedding = layers.Embedding(
                num_users_,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.user_bias = layers.Embedding(num_users_, 1)
            self.movie_embedding = layers.Embedding(
                num_movies_,
                embedding_size,
                embeddings_initializer="he_normal",
                embeddings_regularizer=keras.regularizers.l2(1e-6),
            )
            self.movie_bias = layers.Embedding(num_movies_, 1)

        def call(self, inputs):
            user_vector = self.user_embedding(inputs[:, 0])
            user_bias = self.user_bias(inputs[:, 0])
            movie_vector = self.movie_embedding(inputs[:, 1])
            movie_bias = self.movie_bias(inputs[:, 1])
            dot_user_movie = tf.tensordot(user_vector, movie_vector, 2)
            # Add all the components (including bias)
            x = dot_user_movie + user_bias + movie_bias
            # The sigmoid activation forces the rating to between 0 and 1
            return tf.nn.sigmoid(x)

    model = RecommenderNet(num_users, num_movies, embeddingSize)
    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
    )

    nEpochs = 10
    batchSize = 64
    history = model.fit(
        x=x_train,
        y=y_train,
        batch_size=batchSize,
        epochs=nEpochs,
        verbose=1,
        validation_data=(x_val, y_val),
    )

    weights = model.layers[0].get_weights()[0]
    lenweights = len(weights)
    friendshipMatrix = np.ones((lenweights, lenweights))
    for i in range(lenweights - 1):
        for j in range(i + 1, lenweights):
            interim = np.dot(weights[i], weights[j])
            friendshipMatrix[i][j] = max(interim, 0)
            friendshipMatrix[j][i] = max(interim, 0)

    pd.DataFrame(friendshipMatrix).to_csv(path_input + "/userWeightsFromCF.csv", index=False, header=None)

    W_raw = friendshipMatrix
    # Collaborative filtering run is here #

    # cluster users further
    user2user = np.dot(W_raw, W_raw.T)

    kmeans = KMeans(n_clusters=numClust, random_state=0).fit(user2user)
    centers = kmeans.cluster_centers_

    clusters = np.zeros(len(W_raw))
    for i in range(len(W_raw)):
        clusters[i] = np.atleast_1d(np.argmin(np.linalg.norm(centers - user2user[i], axis=1)))[0]

    users_to_be_filtered["newCluster"] = clusters.T.astype(int)

    newW = np.zeros((numClust, numClust))
    for i in range(numClust):
        for j in range(len(users_to_be_filtered)):
            newW[users_to_be_filtered['newCluster'][j], i] += np.dot(W_raw[users_to_be_filtered['cluster'][j]],
                                                                     W_raw[i].T)

    # add sparsity if applicable
    if 0 < sparsity < numClust:
        SparserW = newW.copy()
        nn = len(newW)
        for i in range(nn):
            similarity = sorted(newW[i], reverse=True)
            threshold = similarity[sparsity - 1]
            for j in range(nn):
                if newW[i][j] <= threshold:
                    SparserW[i][j] = 0
        newW = SparserW

    # normalize by columns
    colNorms = np.linalg.norm(newW, axis=0, ord=1).astype(float)
    newW2 = newW.astype(float)
    for i in range(len(newW[0])):
        if colNorms[i] != 0:
            newW2[:, i] = newW[:, i] / colNorms[i]

    W = newW2

    users_to_be_filtered['cluster'] = users_to_be_filtered['newCluster']
    users_to_be_filtered = users_to_be_filtered.drop('newCluster', axis=1)

    filtered_data_original['reward'] = np.where(filtered_data_original['rating'] < 5, 0 ,1)
    filtered_data_original = filtered_data_original.drop('rating', axis=1)
    filtered_data_original = filtered_data_original.sort_values(by=['timestamp'], ascending=True)
    filtered_data_original = filtered_data_original.reset_index(drop=True)

    # filtered_data_original will have user_id, movie_id, timestamp and reward columns
    return filtered_data_original, top_movies_features, users_to_be_filtered, W
