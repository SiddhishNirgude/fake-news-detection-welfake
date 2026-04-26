"""
preprocess.py
-------------
Preprocessing pipeline for the WELFake fake news detection dataset.
Includes functions for text cleaning, deduplication, combining fields,
and splitting data into train, validation, and test sets.

Author: Siddhish Nirgude
Course: CMSE 928 -- Applied Machine Learning
"""

import re
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def load_data(filepath):
    """
    Load the WELFake CSV file and drop the unnamed index column.

    Parameters
    ----------
    filepath : str
        Path to the raw WELFake_Dataset.csv file.

    Returns
    -------
    pd.DataFrame
        Dataframe with columns: title, text, label.
    """
    df = pd.read_csv(filepath)
    df.drop(columns=["Unnamed: 0"], errors="ignore", inplace=True)
    print(f"Loaded {len(df):,} rows from {filepath}")
    return df


def standardize_nulls(df):
    """
    Strip leading and trailing whitespace from text columns and
    replace empty strings with NaN for consistent null handling.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe with title and text columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with standardized null values.
    """
    for col in ["title", "text"]:
        df[col] = df[col].astype(str).str.strip()
        # 'nan' is the string representation produced by astype(str) on NaN values
        df[col] = df[col].replace({"": np.nan, "nan": np.nan})
    return df


def combine_title_text(df):
    """
    Combine the title and text columns into a single content column.
    Handles missing values by falling back to whichever field is available.
    Rows where both title and text are null are dropped.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with title and text columns.

    Returns
    -------
    pd.DataFrame
        Dataframe with new content column added.
        Rows where both title and text are null are removed.
    """
    before = len(df)
    title_part = df["title"].fillna("")
    text_part = df["text"].fillna("")
    df["content"] = (title_part + " " + text_part).str.strip()

    # Rows where both fields were null produce an empty content string
    df = df[df["content"] != ""].reset_index(drop=True)
    dropped = before - len(df)
    print(f"combine_title_text: dropped {dropped} rows where both title and text were null.")
    return df


def deduplicate(df):
    """
    Remove duplicate rows based on the text column.
    Deduplication is performed before train/test splitting to
    prevent data leakage between splits.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe after combining title and text.

    Returns
    -------
    pd.DataFrame
        Deduplicated dataframe.
    int
        Number of duplicate rows removed.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["text"], keep="first").reset_index(drop=True)
    after = len(df)
    n_removed = before - after
    print(f"Removed {n_removed:,} duplicate rows. Remaining: {after:,}")
    return df, n_removed


def clean_text(text):
    """
    Clean a single text string by applying a sequence of
    normalization steps.

    Steps applied in order:
    1. Lowercase
    2. Remove URLs (http and https)
    3. Remove email addresses
    4. Remove non-ASCII characters
    5. Remove punctuation except apostrophes
    6. Collapse multiple whitespace into single space

    Parameters
    ----------
    text : str
        Raw input text string.

    Returns
    -------
    str
        Cleaned text string. Returns empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+", "", text)
    text = re.sub(r"\S+@\S+", "", text)
    # Encode to ASCII bytes ignoring non-ASCII, then decode back to string
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^\w\s']", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def apply_cleaning(df):
    """
    Apply the clean_text function to the content column of the dataframe.
    Drops rows where the cleaned content is under 10 characters.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe with content column.

    Returns
    -------
    pd.DataFrame
        Dataframe with cleaned content column.
        Rows with content under 10 characters are removed.
    """
    df["content"] = df["content"].apply(clean_text)
    before = len(df)
    df = df[df["content"].str.len() >= 10].reset_index(drop=True)
    after = len(df)
    print(f"Dropped {before - after} rows with content under 10 characters.")
    return df


def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """
    Split the dataframe into train, validation, and test sets using
    stratified sampling to preserve class balance.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with content and label columns.
    test_size : float
        Proportion of data for the test set. Default 0.15.
    val_size : float
        Proportion of data for the validation set. Default 0.15.
    random_state : int
        Random seed for reproducibility. Default 42.

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, val_df, test_df)
    """
    train_val_df, test_df = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=random_state
    )
    # Adjust val proportion relative to the remaining train+val pool
    adjusted_val_size = val_size / (1.0 - test_size)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=adjusted_val_size,
        stratify=train_val_df["label"],
        random_state=random_state,
    )

    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        balance = split["label"].value_counts(normalize=True).round(4).to_dict()
        print(f"{name:5s}: {len(split):,} rows | Fake={balance.get(0, 0):.2%}  Real={balance.get(1, 0):.2%}")

    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df, output_dir):
    """
    Save train, validation, and test dataframes as CSV files.

    Parameters
    ----------
    train_df : pd.DataFrame
    val_df : pd.DataFrame
    test_df : pd.DataFrame
    output_dir : str
        Directory path where CSV files will be saved.

    Returns
    -------
    None
    """
    os.makedirs(output_dir, exist_ok=True)
    splits = {
        "train_clean.csv": train_df,
        "val_clean.csv":   val_df,
        "test_clean.csv":  test_df,
    }
    for fname, split_df in splits.items():
        fpath = os.path.join(output_dir, fname)
        split_df.to_csv(fpath, index=False)
        print(f"Saved {fname}: {len(split_df):,} rows -> {fpath}")


def run_pipeline(filepath, output_dir):
    """
    Run the complete preprocessing pipeline end to end.

    Parameters
    ----------
    filepath : str
        Path to raw CSV file.
    output_dir : str
        Path to save processed CSV files.

    Returns
    -------
    tuple of pd.DataFrame
        (train_df, val_df, test_df)
    """
    df = load_data(filepath)
    rows_load = len(df)

    df = standardize_nulls(df)
    df = combine_title_text(df)
    rows_after_combine = len(df)

    df, _ = deduplicate(df)
    rows_after_dedup = len(df)

    df = apply_cleaning(df)
    rows_after_clean = len(df)

    train_df, val_df, test_df = split_data(df)
    save_splits(train_df, val_df, test_df, output_dir)

    print()
    print("=" * 52)
    print(f"{'Step':<20} {'Rows In':>10} {'Rows Out':>10}")
    print("=" * 52)
    print(f"{'Load':<20} {rows_load:>10,} {rows_load:>10,}")
    print(f"{'Combine fields':<20} {rows_load:>10,} {rows_after_combine:>10,}")
    print(f"{'Deduplicate':<20} {rows_after_combine:>10,} {rows_after_dedup:>10,}")
    print(f"{'Clean text':<20} {rows_after_dedup:>10,} {rows_after_clean:>10,}")
    print(f"{'Train split':<20} {rows_after_clean:>10,} {len(train_df):>10,}")
    print(f"{'Val split':<20} {rows_after_clean:>10,} {len(val_df):>10,}")
    print(f"{'Test split':<20} {rows_after_clean:>10,} {len(test_df):>10,}")
    print("=" * 52)

    return train_df, val_df, test_df
