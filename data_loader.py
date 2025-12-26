import pandas as pd

class MoviesLensLoader:
    def __init__(self, base_path="data/raw/ml-100k/"):
        self.base = base_path

    def load_rating(self):
        cols = ["user_id", "item_id", "rating", "timestamp"]
        return pd.read_csv(f"{self.base}u.data", sep="\t", names=cols)

    def load_items(self):
        cols = (
            ["item_id", "title", "release_date", "video_release", "imdb_url"]
            + [f"genre_{i}" for i in range(19)]
        )
        return pd.read_csv(
            f"{self.base}u.item",
            sep="|",
            names=cols,
            encoding="latin-1",
        )

    def load_all(self):
        return self.load_rating(), self.load_items()
