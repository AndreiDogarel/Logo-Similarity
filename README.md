# Logo Similarity using ResNet50 and DBSCAN

## Introduction
This project groups website logos based on visual similarity using a pre-trained ResNet50 model for feature extraction and DBSCAN for clustering. The goal is to analyze and classify logos, identifying visually similar ones, even if they belong to different domains.

## Approach

### 1. **Downloading Favicons**
A list of websites is processed to fetch their favicons. The `FaviconExtractor` extracts these favicons and saves them for further processing.

### 2. **Preprocessing Images**
The `ImagePreprocessor` resizes images to a standard shape (32x32 pixels) while maintaining color information. This step ensures consistency before feature extraction.

### 3. **Feature Extraction**
We use a pre-trained **ResNet50** (from PyTorch) to extract deep feature representations of each logo. The `FeatureExtractor` processes each image and saves the feature vectors as `.pt` files.

### 4. **Storing Features**
The `FeatureStorage` module consolidates all extracted features into a single `.parquet` file. This format allows for efficient storage and retrieval of high-dimensional data.

### 5. **Grouping Logos by Similarity**
Using **DBSCAN**, we cluster logos into groups based on feature similarity. The `LogoGrouper` assigns logos to clusters, ensuring that visually similar images are placed together.

## Installation
To set up the project, install the necessary dependencies:

```sh
pip install -r requirements.txt
```

Ensure you have **Python 3.12+** installed and that `pip` is up to date.

## Running the Pipeline
The entire pipeline is executed through the `LogoSimilarityPipeline` class. Run the following command to process and group the logos:

```sh
python main.py
```

## Output
The program outputs multiple groups of similar logos in a JSON file (`logo_clusters.json`). Each group contains websites whose logos are visually similar. If a logo is unique, it forms its own group.

## Future Improvements
- Fine-tune the ResNet50 model on a logo dataset for better feature extraction.
- Optimize clustering parameters (`eps` and `min_samples` in DBSCAN) for more accurate grouping.
- Expand support for other image formats and logo extraction techniques.