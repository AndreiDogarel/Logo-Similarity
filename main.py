import os
from data_extraction.FaviconDownloader import FaviconDownloader
from preprocessing.ImagePreprocessor import ImagePreprocessor
from feature_extraction.FeatureExtractor import FeatureExtractor
from grouping.FeatureStorage import FeatureStorage
from grouping.LogoGrouper import LogoGrouper

class LogoSimilarityPipeline:
    def __init__(self):
        self.downloader = FaviconDownloader("data_extraction\\logos.snappy.parquet")
        self.preprocessor = ImagePreprocessor("favicons")
        self.model = FeatureExtractor()
        self.storage = FeatureStorage()
        self.grouper = LogoGrouper(self.storage)

    def run(self):
        self.downloader.run()
        self.preprocessor.preprocess_images()
        self.model.process_directory("favicons_preprocessed")
        self.storage.load_features()
        self.grouper.group_logos()

if __name__ == "__main__":
    runner = LogoSimilarityPipeline()
    runner.run()