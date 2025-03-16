import numpy as np
import json
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


class LogoGrouper:
    def __init__(self, feature_storage, output_file="logo_clusters.json", eps=0.5, min_samples=2):
        self.feature_storage = feature_storage
        self.output_file = output_file
        self.eps = eps
        self.min_samples = min_samples

    def group_logos(self):
        df = self.feature_storage.load_from_parquet()
        features = np.stack(df["features"].values)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        clustering = DBSCAN(eps=self.eps, min_samples=self.min_samples, metric="cosine").fit(features_scaled)

        clusters = {}
        for filename, label in zip(df["filename"], clustering.labels_):
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(filename)

        with open(self.output_file, "w") as f:
            json.dump({str(k): v for k, v in clusters.items()}, f, indent=4)
        print(f"Saved clustering results to {self.output_file}")
