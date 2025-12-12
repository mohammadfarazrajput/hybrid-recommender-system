import pandas as pd

class MoviesLensLoader():
    def __init__(self):
        self.base = "data/raw/ml-100k/"
    
    def load_rating(self):
        columns = ["user_id", "item_id", "rating", "timestamp"]
        ratings = pd.read_csv(f"{self.base}u.data", sep="\t", names=columns)
        return ratings
    
    def load_items(self):
        columns = ["item_id", "title", "release_date", "video_release", "imdb_url"] + [f"genre_{i}" for i in range(19)]
        items = pd.read_csv(f"{self.base}u.item", sep="|", names=columns, encoding="latin-1")
        return items
    
    def load_all(self):
        ratings = self.load_rating()
        items = self.load_items()
        return ratings, items
 