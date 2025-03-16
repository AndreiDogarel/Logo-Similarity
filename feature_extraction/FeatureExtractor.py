import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

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
