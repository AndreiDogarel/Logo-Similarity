import torch
import pandas as pd
from pathlib import Path

class FeatureStorage:
    def __init__(self, input_dir="features", output_file="features.parquet"):
        self.input_dir = Path(input_dir)
        self.output_file = output_file

    def load_features(self):
        data = []
        for file in self.input_dir.glob("*.pt"):
            vector = torch.load(file).squeeze().tolist()
            data.append({"filename": file.stem, "features": vector})

        df = pd.DataFrame(data)
        df.to_parquet(self.output_file, index=False)
        print(f"Saved {len(data)} feature vectors to {self.output_file}")

    def load_from_parquet(self):
        return pd.read_parquet(self.output_file)
