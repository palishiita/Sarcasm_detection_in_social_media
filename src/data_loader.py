import pandas as pd

def load_semeval(path):
    return pd.read_csv(
        path,
        sep="\t",
        skiprows=1,
        header=None,
        names=["index", "label", "text"]
    )

def load_reddit(path):
    return pd.read_csv(path)

def load_news(path):
    return pd.read_json(path, lines=True)

def load_all(config):
    semeval = load_semeval(config["semeval_path"])
    reddit = load_reddit(config["reddit_path"])
    news = load_news(config["news_path"])
    return semeval, reddit, news

def load_processed(path):
    return pd.read_csv(path)