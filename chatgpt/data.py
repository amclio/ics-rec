import sys
import os
import shutil
import pandas as pd
import numpy as np
from recommenders.utils.timer import Timer
from recommenders.datasets import movielens
from recommenders.datasets.python_splitters import python_chrono_split
from recommenders.evaluation.python_evaluation import (
    rmse,
    mae,
    rsquared,
    exp_var,
    map_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    get_top_k_items,
)

train_ratio = 0.8

# Select MovieLens data size: 100k, 1m, 10m, or 20m
# MOVIELENS_DATA_SIZE = "100k"

# ratings_transformed_df = movielens.load_pandas_df(
#     size=MOVIELENS_DATA_SIZE, header=["userID", "itemID", "rating", "timestamp"]
# )

# ratings_df = ratings_transformed_df


movielens_dir = "assets/datasets/movielens-small"
movies_file = os.path.join(movielens_dir, "movies.csv")
ratings_file = os.path.join(movielens_dir, "ratings.csv")

movies_df = pd.read_csv(movies_file)
ratings_df = pd.read_csv(ratings_file)

user_id_list = ratings_df["userId"].unique().tolist()

# train, test = python_chrono_split(ratings_transformed_df, 0.75)

shuffled_ratings_df = ratings_df.sample(frac=1, random_state=42)
train_size = int(len(ratings_df) * train_ratio)

train = shuffled_ratings_df[:train_size]
test = shuffled_ratings_df[train_size:]
