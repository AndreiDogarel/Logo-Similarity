# Logo Similarity using ResNet50 and DBSCAN

## Introduction
This project groups website logos based on visual similarity using a pre-trained ResNet50 model for feature extraction and DBSCAN for clustering. The goal is to analyze and classify logos, identifying visually similar ones, even if they belong to different domains.

## Approach

### 1. **Downloading Favicons**
```python
class FaviconDownloader:
    GOOGLE_FAVICON_API = "https://www.google.com/s2/favicons?sz=64&domain={}"
    
    def __init__(self, parquet_file, output_dir="favicons"):
        self.parquet_file = parquet_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def get_domains(self):
        try:
            table = pq.read_table(self.parquet_file)
            df = table.to_pandas()
            if "domain" not in df.columns:
                raise ValueError("No domains found")
            return df["domain"].dropna().unique()
        except Exception as e:
            print(f"Error while reading Parquet file {e}")
            return []
    
    def download_favicon(self, domain):
        url = self.GOOGLE_FAVICON_API.format(domain)
        try:
            response = requests.get(url, stream=True)
            if response.status_code == 200:
                filepath = os.path.join(self.output_dir, f"{domain}.png")
                with open(filepath, "wb") as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Saved: {filepath}")
            else:
                print(f"Error ({response.status_code}) for {domain}")
        except Exception as e:
            print(f"Error while downloading icon for {domain}: {e}")
    
    def run(self):
        domains = self.get_domains()
        for domain in domains:
            self.download_favicon(domain)
```
**FaviconDownloader - Downloading Website Logos**
This class downloads favicons (website logos) from a list of domains provided in a `.parquet` file.  
It uses the Google Favicon API to retrieve images and saves them in the `favicons` directory.

#### **Attributes:**
- `input_file (str)`: Path to the `.parquet` file containing website domains.  
- `output_dir (Path)`: Directory where downloaded favicons are stored.  

#### **Methods:**
- `__init__(self, input_file, output_dir="favicons")`: Initializes the class with the input file and output directory.  
- `download_favicon(self, domain)`: Downloads the favicon for a given domain and saves it as a PNG file.  
- `run(self)`: Reads the `.parquet` file and downloads favicons for all domains.

### 2. **Preprocessing Images**
```python
class ImagePreprocessor:
    def __init__(self, input_dir, output_dir="favicons_preprocessed", size: tuple = (32, 32)):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.size = size
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def preprocess_images(self):
        for file in self.input_dir.glob("*.png"):
            self.process_image(file)

    def process_image(self, image_path: Path):
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            print(f"Failed to read {image_path}")
            return

        img_resized = cv2.resize(img, self.size, interpolation=cv2.INTER_AREA)
        output_path = self.output_dir / image_path.name
        cv2.imwrite(str(output_path), img_resized)
        print(f"Processed: {image_path} -> {output_path}")
```
**ImagePreprocessor - Image Processing**
This class processes favicon images by resizing them to a standard size before feature extraction.  
It ensures that all images have the same resolution for consistency in machine learning models.

#### **Attributes:**
- `input_dir (Path)`: Directory containing the original favicon images.  
- `output_dir (Path)`: Directory where the resized images will be saved.  
- `size (tuple)`: Target size for resizing the images.  

#### **Methods:**
- `__init__(self, input_dir, output_dir="favicons_preprocessed", size=(32, 32))`: Initializes the class with input/output directories and image size.  
- `preprocess_images(self)`: Reads each image from `input_dir`, resizes it, and saves it to `output_dir`.  

### 3. **Feature Extraction**
```python
class FeatureExtractor:
    def __init__(self, model_name="resnet50", device=None):
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_name)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_name):
        if model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        else:
            raise ValueError("Unsupported model")
        
        model = nn.Sequential(*list(model.children())[:-1])
        model.eval()
        return model.to(self.device)
    
    def extract_features(self, image_path):
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image)
        
        return features.squeeze().cpu().numpy()
    
    def process_directory(self, input_dir, output_dir="features"):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for image_path in input_dir.glob("*.png"):
            features = self.extract_features(image_path)
            torch.save(torch.tensor(features), output_dir / f"{image_path.stem}.pt")
            print(f"Extracted features for {image_path.name}")
```
**FeatureExtractor - Extracting Features Using ResNet50**
This class extracts **feature vectors** from images using a pre-trained **ResNet50** model.  
It converts each image into a high-dimensional vector representation.

#### **Attributes:**
- `model (torch.nn.Module)`: A modified ResNet50 model without the fully connected (FC) layer.  
- `transform (torchvision.transforms.Compose)`: A set of transformations applied to the image before feeding it into ResNet50.  

#### **Methods:**
- `__init__(self)`: Initializes the ResNet50 model and preprocessing transformations.  
- `extract_features(self, image_path)`: Loads an image, applies transformations, and extracts a feature vector.  
- `process_directory(self, input_dir, output_dir="features")`: Reads all images from `input_dir`, extracts features, and saves them as `.pt` files in `output_dir`.  


### 4. **Storing Features**
```python
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
```
**FeatureStorage - Storing Extracted Features**
This class manages the storage of feature vectors by **saving and loading** them in **Parquet format**.  
It allows efficient access to precomputed features.

#### **Attributes:**
- `input_dir (Path)`: Directory containing `.pt` files with extracted features.  
- `output_file (str)`: Path to the `.parquet` file where features will be stored.  

#### **Methods:**
- `__init__(self, input_dir="features", output_file="features.parquet")`: Initializes the class with input directory and output file.  
- `save_to_parquet(self)`: Loads feature vectors from `.pt` files and saves them into a `.parquet` file.  
- `load_from_parquet(self)`: Reads feature vectors from the `.parquet` file and returns a DataFrame.  

### 5. **Grouping Logos by Similarity**
```python
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
```
**LogoGrouper - Grouping Logos Using DBSCAN**
This class groups logos based on **cosine similarity** using the **DBSCAN clustering algorithm**.  
It reads feature vectors from the `.parquet` file and assigns each logo to a cluster.

#### **Attributes:**
- `feature_storage (FeatureStorage)`: An instance of `FeatureStorage` to retrieve stored feature vectors.  
- `output_file (str)`: Path to the `.json` file where clustering results are saved.  
- `eps (float)`: Maximum distance between two points to be considered in the same cluster.  
- `min_samples (int)`: Minimum number of points required to form a cluster.  

#### **Methods:**
- `__init__(self, feature_storage, output_file="logo_clusters.json", eps=0.5, min_samples=2)`: Initializes the class with clustering parameters and file paths.  
- `group_logos(self)`: Loads feature vectors, applies DBSCAN, and saves the clustering results in a `.json` file.  

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
The program outputs multiple groups of similar logos in a JSON file ([View Clustering Results](logo_clusters.json)). Each group contains websites whose logos are visually similar. If a logo is unique, it forms its own group.


## Future Improvements
- Fine-tune the ResNet50 model on a logo dataset for better feature extraction.
- Optimize clustering parameters (`eps` and `min_samples` in DBSCAN) for more accurate grouping.
- Expand support for other image formats and logo extraction techniques.