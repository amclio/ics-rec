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


class UserInstance:
    def __init__(self, user_id) -> None:
        self.user_id = user_id

    def predict(self):
        user_id = self.user_id

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

        self.movies_watched_by_user = movies_watched_by_user
        self.movies_not_watched = movies_not_watched
        self.ratings = ratings

    def get_top_k_from_recommended(self, k):
        # NOTE: Indices are returned from argsort()

        ratings = self.ratings

        return ratings.argsort()[-k:][::-1]
