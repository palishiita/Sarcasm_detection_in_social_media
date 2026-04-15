import pandas as pd

def load_spirs(path):
    return pd.read_csv(path)

def load_reddit(path):
    return pd.read_csv(path)

def load_news(path):
    return pd.read_json(path, lines=True)

def load_all(config):
    spirs = load_spirs(config["spirs_path"])
    reddit = load_reddit(config["reddit_path"])
    news = load_news(config["news_path"])
    return spirs, reddit, news