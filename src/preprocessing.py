import re

def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", "", text)
    return text

def preprocess_dataframe(df, text_column):
    df[text_column] = df[text_column].apply(clean_text)
    return df