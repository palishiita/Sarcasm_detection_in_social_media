import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split


def clean_text(text):
    text = str(text)
    text = re.sub(r"http\S+|www\S+", "", text)      # remove URLs
    text = re.sub(r"@\w+", "", text)                 # remove mentions
    text = re.sub(r"#(\w+)", r"\1", text)            # strip # but keep word
    text = re.sub(r"[^\x00-\x7F]+", " ", text)      # remove non-ASCII
    text = re.sub(r"\s+", " ", text).strip()         # normalise whitespace
    return text.lower()


def preprocess_dataframe(df, text_column):
    df = df.copy()
    df[text_column] = df[text_column].apply(clean_text)
    return df


def drop_empty(df, text_column, label_column="label"):
    df = df.dropna(subset=[text_column, label_column])
    df = df[df[text_column].str.strip() != ""]
    df[text_column] = df[text_column].fillna("").astype(str)  # catch any remaining NaNs
    return df.reset_index(drop=True)


def split_and_save(df, output_dir, name, test_size=0.2, random_state=42):
    """Stratified train/test split and save to processed folder."""

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df["label"],
        random_state=random_state
    )

    os.makedirs(output_dir, exist_ok=True)
    df.to_csv(os.path.join(output_dir, f"{name}_clean.csv"), index=False)
    train_df.to_csv(os.path.join(output_dir, f"{name}_train.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, f"{name}_test.csv"), index=False)

    print(f"{name} → train: {len(train_df)}, test: {len(test_df)}")
    return train_df, test_df