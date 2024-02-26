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


def load_data(path, dataset_name):
    """
    Loads data from a CSV file into a Pandas DataFrame and allows user to remove specified columns.
    This function is specific to either the primary or secondary dataset as indicated by the dataset_name parameter.
    """
    df = pd.read_csv(path)
    print(f"Columns available in the {dataset_name} dataset: ", df.columns.tolist())
    cols_to_remove = input(f"Enter the columns you wish to remove from the {dataset_name} dataset (separated by commas): ").split(',')
    cols_to_remove = [col.strip() for col in cols_to_remove if col.strip() in df.columns]  # Validate input
    if cols_to_remove:
        df.drop(cols_to_remove, axis=1, inplace=True)
    return df

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

def merge_info(primary_info, secondary_info):
    if pd.isna(primary_info) or primary_info == '':
        return secondary_info  # Use secondary if primary is missing
    return primary_info  # Prefer primary information

def main():
    # Ask user to input the two datasets
    print("For the primary dataset:")
    path_primary = input("Enter the path to the primary dataset (CSV format): ")
    df_primary = load_data(path_primary, 'primary')
    
    print("\nFor the secondary dataset:")
    path_secondary = input("Enter the path to the secondary dataset (CSV format): ")
    df_secondary = load_data(path_secondary, 'secondary')

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

    # Prepare to store matched records
    matched_records = []
    primary_match_counts = {}

    # Automatically detect shared and unique fields
    shared_fields = set(df_primary.columns).intersection(set(df_secondary.columns))
    unique_primary_fields = set(df_primary.columns) - shared_fields
    unique_secondary_fields = set(df_secondary.columns) - shared_fields

    with open('match_results.txt', 'w') as results_file:
        for index, secondary_row in df_secondary.iterrows():
            combined_row_string = df_secondary_resized.iloc[index]['combined']  # Assuming 'combined' column exists
            closest_match_id = find_closest_match_id_tf_idf(combined_row_string, loaded_vectorizer, vectors)
            if closest_match_id is not None:
                primary_row = df_primary.iloc[closest_match_id]

                # Increment the match count for this primary record
                primary_match_counts[closest_match_id] = primary_match_counts.get(closest_match_id, 0) + 1

                # Merge shared fields and retain unique fields from both datasets
                combined_record = {}
                for field in shared_fields:
                    combined_record[field] = merge_info(primary_row[field], secondary_row[field])
                for field in unique_primary_fields:
                    combined_record[f'primary_{field}'] = primary_row[field]
                for field in unique_secondary_fields:
                    combined_record[f'secondary_{field}'] = secondary_row[field]

                matched_records.append(combined_record)

                # Write to the text file
                results_file.write(f"Closest match for record {index} (Secondary): {combined_row_string}\n")
                results_file.write(f"Is matched with (Primary): {primary_row.to_dict()}\n\n")
            else:
                results_file.write(f"Record {index} (Secondary): {combined_row_string} has no appropriate match\n\n")

    multiple_matches_count = sum(1 for count in primary_match_counts.values() if count > 1)

    summary_filename = 'summary_info.txt'
    total_matches = len(matched_records)
    unique_primary_count = len(unique_primary_fields)
    unique_secondary_count = len(unique_secondary_fields)
    shared_field_count = len(shared_fields)

    with open(summary_filename, 'w') as summary_file:
        summary_file.write(f"Total number of matched records: {total_matches}\n")
        summary_file.write(f"Number of unique fields from primary dataset: {unique_primary_count}\n")
        summary_file.write(f"Number of unique fields from secondary dataset: {unique_secondary_count}\n")
        summary_file.write(f"Number of shared fields between datasets: {shared_field_count}\n")
        summary_file.write(f"Number of primary records matched more than once: {multiple_matches_count}\n")
    ...


    if matched_records:
        matched_df = pd.DataFrame(matched_records)
        matched_df.to_csv('matched_records.csv', index=False)
        print("Matched records have been saved to matched_records.csv")
    else:
        print("No matches found.")

    print("The match results have been saved to match_results.txt")

if __name__ == "__main__":
    main()
