# NOTE: Implementing Precision/Recall@K

import tensorflow as tf
from data import user2user_encoded, user_ids
from predict import UserInstance
from tensorflow import keras

# NOTE: Top-K constant
K = 10
# threshold_rating = 3.5
threshold_rating = 0

precision_list = []
recall_list = []


for i, user_id in enumerate(user_ids):
    instance = UserInstance(user_id)
    instance.predict()

    recommended_movie_ids = instance.get_top_k_from_recommended(K)

    movies_in_high_rating = instance.movies_watched_by_user[
        instance.movies_watched_by_user["rating"] >= threshold_rating
    ]

    intersection = movies_in_high_rating[
        movies_in_high_rating["movieId"].isin(recommended_movie_ids)
    ]
    intersection_count = len(intersection.index)

    precision_at_k = intersection_count / K
    recall_at_k = intersection_count / len(movies_in_high_rating.index)

    precision_list.append(precision_at_k)
    recall_list.append(recall_at_k)

    print(
        "UserID: {} Precision@K: {} Recall@K: {}".format(
            user_id, precision_at_k, recall_at_k
        )
    )

print(
    "Precision@K Average: {}".format(
        sum(prec for i, prec in enumerate(precision_list)) / len(precision_list)
    )
)

print(
    "Recall@K Average: {}".format(
        sum(rec for i, rec in enumerate(recall_list)) / len(recall_list)
    )
)
