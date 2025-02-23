import pandas as pd
from PIL import Image
import requests
from pathlib import Path
import logging
from tqdm import tqdm
import io
from urllib.parse import urlparse

class ImageProcessor:
    def __init__(
        self, 
        csv_path: str, 
        output_dir: str = "processed_images",
        url_column: str = "img",
        target_size: tuple = (256,256)
    ):
        """
        Initialize the image processor.
        
        Args:
            csv_path: Path to the CSV file
            output_dir: Directory to save processed images
            url_column: Name of the column containing image URLs
            target_size: Tuple of (width, height) for resizing
        """
        self.csv_path = Path(csv_path)
        self.output_dir = Path(output_dir)
        self.url_column = url_column
        self.target_size = target_size
        
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            filename=self.output_dir / 'image_processing.log'
        )
        
    def get_image_from_url(self, url: str) -> Image.Image:
        """Download image from URL and return as PIL Image object."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return Image.open(io.BytesIO(response.content))
        except requests.exceptions.RequestException as e:
            logging.error(f"Error downloading image from {url}: {e}")
            raise
        except Exception as e:
            logging.error(f"Error processing image from {url}: {e}")
            raise

    def resize_image(self, image: Image.Image) -> Image.Image:
        """Resize image while maintaining aspect ratio."""
        try:
            image.thumbnail(self.target_size, Image.Resampling.LANCZOS)
            return image
        except Exception as e:
            logging.error(f"Error resizing image: {e}")
            raise

    def process_images(self):
        """Process all images from the CSV file."""
        try:
            
            df = pd.read_csv(self.csv_path)
            
            if self.url_column not in df.columns or 'id' not in df.columns:
                raise ValueError(f"Required columns not found in CSV file")
            
            
            successful = 0
            failed = 0
            
            
            for index, row in tqdm(df.iterrows(), total=len(df), desc="Processing images"):
                url = row[self.url_column]
                
                if pd.isna(url):
                    logging.warning(f"Row {index}: Empty URL")
                    failed += 1
                    continue
                
                try:
                    
                    image = self.get_image_from_url(url)
                    processed_image = self.resize_image(image)
                    
                    
                    filename = f"{row['id']}.jpg"
                    output_path = self.output_dir / filename
                    
                    
                    processed_image.save(output_path, format='JPEG')
                    
                    successful += 1
                    
                except Exception as e:
                    logging.error(f"Row {index}: Failed to process image from {url}: {e}")
                    failed += 1
                    continue
            
            
            logging.info(f"Processing complete. Successful: {successful}, Failed: {failed}")
            print(f"\nProcessing complete. Successful: {successful}, Failed: {failed}")
            print(f"Check {self.output_dir} for processed images")
            print(f"Check {self.output_dir}/image_processing.log for detailed logs")
            
        except Exception as e:
            logging.error(f"Error processing CSV file: {e}")
            raise

def main():
    
    processor = ImageProcessor(
        csv_path="data/opal_auction_data_20250222_214831.csv",  
        output_dir="processed_images",
        url_column="img",              
        target_size=(256,256)         
    )
    processor.process_images()

if __name__ == "__main__":
    main()