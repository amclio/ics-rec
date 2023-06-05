import numpy as np
import pandas as pd

MOVIE_PATH = "assets/datasets/movielens-small/movies.csv"
RATING_PATH = "assets/datasets/movielens-small/ratings.csv"

RATING_THRESHOLD = 5


rating_df = pd.read_csv(RATING_PATH)
movies_df = pd.read_csv(MOVIE_PATH)


def get_genres(df=movies_df):
    # 2. Extract the column 'genres' from the Dataframe
    genres = df["genres"]

    # 3. Transform the extracted column into the list
    genres_list = genres.tolist()

    # 4. Each element in the list will be the string that contains two words divided by the character '|'. Separate that string
    split_genres_list = [genre.split("|") for genre in genres_list]

    # 5. Create new list that contains separated string. Each string element should be unique.
    unique_genres_list = list(
        set([item for sublist in split_genres_list for item in sublist])
    )

    return unique_genres_list


high_rating_movies = rating_df[
    (rating_df["userId"] == 1) & (rating_df["rating"] >= RATING_THRESHOLD)
]["movieId"]

merged_ratings = pd.merge(left=high_rating_movies, right=movies_df, on="movieId")

filtered_genres = get_genres(merged_ratings)
all_genres = get_genres(movies_df)

print(filtered_genres)
