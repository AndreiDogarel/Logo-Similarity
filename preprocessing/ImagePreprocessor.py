import cv2
from pathlib import Path

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