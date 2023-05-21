import os

import numpy as np
import pandas as pd
import tensorflow as tf
from data import (
    df,
    movie2movie_encoded,
    movie_encoded2movie,
    movielens_dir,
    user2user_encoded,
)

model = tf.keras.models.load_model(
    "collaborative-filtering/models/collaborative-filtering-movielens"
)

movie_list_file = os.path.join(movielens_dir, "movies.csv")
movie_df = pd.read_csv(movie_list_file)

# # Let us get a user and see the top recommendations.
user_id = df.userId.sample(1).iloc[0]
movies_watched_by_user = df[df.userId == user_id]
movies_not_watched = movie_df[
    ~movie_df["movieId"].isin(movies_watched_by_user.movieId.values)
]["movieId"]
movies_not_watched = list(
    set(movies_not_watched).intersection(set(movie2movie_encoded.keys()))
)
# NOTE: Encoded Movie ID
movies_not_watched = [[movie2movie_encoded.get(x)] for x in movies_not_watched]
# NOTE: Encoded User ID.
user_encoder = user2user_encoded.get(user_id)
user_movie_array = np.hstack(
    ([[user_encoder]] * len(movies_not_watched), movies_not_watched)  # type: ignore
)

ratings = model.predict(user_movie_array).flatten()  # type: ignore
top_ratings_indices = ratings.argsort()[-10:][::-1]
recommended_movie_ids = [
    # NOTE: `movie_encoded2movie` is finally used here!
    movie_encoded2movie.get(movies_not_watched[x][0])
    for x in top_ratings_indices
]

# NOTE: Below are the useless codes. It's time to apply confusion matrix.

print("Showing recommendations for user: {}".format(user_id))
print("====" * 9)
print("Movies with high ratings from user")
print("----" * 8)
top_movies_user = (
    movies_watched_by_user.sort_values(by="rating", ascending=False)
    .head(5)
    .movieId.values
)
movie_df_rows = movie_df[movie_df["movieId"].isin(top_movies_user)]
for row in movie_df_rows.itertuples():
    print(row.title, ":", row.genres)

print("----" * 8)
print("Top 10 movie recommendations")
print("----" * 8)
recommended_movies = movie_df[movie_df["movieId"].isin(recommended_movie_ids)]
for row in recommended_movies.itertuples():
    print(row.title, ":", row.genres)
