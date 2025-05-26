from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
import pandas as pd


def split_data(df):

    user_encoder = LabelEncoder()
    movie_encoder = LabelEncoder()
    df["userId"] = user_encoder.fit_transform(df["userId"])
    df["movieId"] = movie_encoder.fit_transform(df["movieId"])

    file_path = f"/opt/models"
    with open(f"{file_path}/user_encoder.pkl", "wb") as f:
        pickle.dump(user_encoder, f)

    with open(f"{file_path}/movie_encoder.pkl", "wb") as f:
        pickle.dump(movie_encoder, f)

    train_list, val_list, test_list = [], [], []

    for user_id, user_df in df.groupby("userId"):
        if len(user_df) < 5:
            # Too few interactions → put all in train
            train_list.append(user_df)
            continue

        train_temp, test = train_test_split(user_df, test_size=0.2, random_state=42)
        train, val = train_test_split(train_temp, test_size=0.1, random_state=42)

        train_list.append(train)
        val_list.append(val)
        test_list.append(test)

    # Step 3: Combine all per-user splits
    train_df = pd.concat(train_list).reset_index(drop=True)
    val_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, val_df, test_df

    # unique_users = df["userId"].unique()
    # train_users, test_users = train_test_split(
    #     unique_users, test_size=0.2, random_state=42
    # )
    # train_users, val_users = train_test_split(
    #     train_users, test_size=0.1, random_state=42
    # )

    # train_df = df[df["userId"].isin(train_users)]
    # val_df = df[df["userId"].isin(val_users)]
    # test_df = df[df["userId"].isin(test_users)]

    # return train_df, val_df, test_df
