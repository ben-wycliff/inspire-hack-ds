#!/usr/bin/env python3

import pickle
from fuzzywuzzy import process
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pandas as pd


def find_similar_columns(df1, df2, threshold=80):
    """
    Identifies similar columns across two dataframes based on column names.
    """
    similar_columns = {}
    for col1 in df1.columns:
        match, score = process.extractOne(col1, df2.columns)
        if score >= threshold:
            similar_columns[col1] = match
    return similar_columns


def load_data(path):
    """
    Loads data from a CSV file into a Pandas DataFrame.
    """
    return pd.read_csv(path)


def combine_columns(df, columns):
    """
    Combines specified columns into a single 'combined' column.
    """
    df['combined'] = df[columns].apply(lambda row: " ".join(row.astype(str)), axis=1)
    return df


def vectorize_text(df, column_name='combined', vectorizer=None):
    """
    Vectorizes the text of the specified column in a dataframe.
    """
    if vectorizer is None:
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(df[column_name])
    else:
        vectors = vectorizer.transform(df[column_name])
    return vectors, vectorizer


def find_closest_match_id_tf_idf(string_row, loaded_vectorizer, vectors, threshold=0.7):
    """
    Finds the closest match for a text row using TF-IDF vectors.
    """
    new_vector = loaded_vectorizer.transform([string_row])
    similarity_scores = cosine_similarity(new_vector, vectors)
    closest_match_index = np.argmax(similarity_scores)
    if similarity_scores[0, closest_match_index] >= threshold:
        return closest_match_index
    return None


def main():
    # Ask user to input the two datasets
    path_primary = input("Enter the path to the primary dataset (CSV format): ")
    path_secondary = input("Enter the path to the secondary dataset (CSV format): ")

    # Load the datasets
    df_primary = load_data(path_primary)
    df_secondary = load_data(path_secondary)

    # Find and resize the datasets based on similar columns
    similar_columns = find_similar_columns(df_primary, df_secondary, threshold=80)
    df_secondary_resized = df_secondary[list(similar_columns.values())]
    df_primary_resized = df_primary[list(similar_columns.keys())]

    # Combine columns into a single 'combined' column for each dataset
    df_primary_resized = combine_columns(df_primary_resized, similar_columns.keys())
    df_secondary_resized = combine_columns(df_secondary_resized, similar_columns.values())

    # Vectorize the 'combined' column of the primary dataset
    vectors, vectorizer = vectorize_text(df_primary_resized)

    # Save the vectorizer for future use
    with open('tf-idf-vectorizer.pkl', 'wb') as file:
        pickle.dump(vectorizer, file)

    # Load the vectorizer and transform the 'combined' column of the secondary dataset
    with open('tf-idf-vectorizer.pkl', 'rb') as file:
        loaded_vectorizer = pickle.load(file)
    secondary_vectors = vectorize_text(df_secondary_resized, vectorizer=loaded_vectorizer)[0]

    # Loop through rows in the secondary dataset to find matches
    for index, row in df_secondary_resized.iterrows():
        combined_row_string = row['combined']
        closest_match_id = find_closest_match_id_tf_idf(combined_row_string, loaded_vectorizer, vectors)
        if closest_match_id is not None:
            closest_match = df_primary_resized.iloc[closest_match_id]
            print(f"Closest match for record {index} (Secondary): {combined_row_string}")
            print(f"Is matched with (Primary): {closest_match['combined']}\n")
        else:
            print(f"Record {index} (Secondary): {combined_row_string} has no appropriate match\n")


if __name__ == "__main__":
    main()
