from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def split_data(df):
    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df["userId"] = user_encoder.fit_transform(df["userId"])
    df["movieId"] = movie_encoder.fit_transform(df["movieId"])

    num_users = df["userId"].nunique()
    num_movies = df["movieId"].nunique()

    unique_users = df["userId"].unique()
    train_users, test_users = train_test_split(
        unique_users, test_size=0.2, random_state=42
    )
    train_users, val_users = train_test_split(
        train_users, test_size=0.1, random_state=42
    )

    train_df = df[df["userId"].isin(train_users)]
    val_df = df[df["userId"].isin(val_users)]
    test_df = df[df["userId"].isin(test_users)]

    return train_df, val_df, test_df
