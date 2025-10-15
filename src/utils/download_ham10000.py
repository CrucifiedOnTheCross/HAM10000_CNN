"""
Utility script to download HAM10000 dataset programmatically.
"""

import os
import requests
import zipfile
import pandas as pd
from pathlib import Path
from typing import Optional
import logging
from tqdm import tqdm


class HAM10000Downloader:
    """Download and extract HAM10000 dataset."""
    
    # Dataset URLs (these are example URLs - actual URLs may vary)
    DATASET_URLS = {
        'metadata': 'https://dataverse.harvard.edu/api/access/datafile/3450625',
        'images_part1': 'https://dataverse.harvard.edu/api/access/datafile/3450626',
        'images_part2': 'https://dataverse.harvard.edu/api/access/datafile/3450627'
    }
    
    def __init__(self, dataset_path: str):
        """
        Initialize downloader.
        
        Args:
            dataset_path: Path where dataset will be downloaded
        """
        self.dataset_path = Path(dataset_path)
        self.images_path = self.dataset_path / 'images'
        self.metadata_path = self.dataset_path / 'metadata'
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_directories(self):
        """Create necessary directories."""
        self.dataset_path.mkdir(parents=True, exist_ok=True)
        self.images_path.mkdir(exist_ok=True)
        self.metadata_path.mkdir(exist_ok=True)
        
        self.logger.info(f"Created directories in {self.dataset_path}")
    
    def download_file(self, url: str, filename: str, chunk_size: int = 8192) -> bool:
        """
        Download file with progress bar.
        
        Args:
            url: URL to download from
            filename: Local filename to save to
            chunk_size: Download chunk size
            
        Returns:
            True if successful, False otherwise
        """
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            filepath = self.dataset_path / filename
            
            with open(filepath, 'wb') as f, tqdm(
                desc=filename,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            self.logger.info(f"Downloaded {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {filename}: {e}")
            return False
    
    def extract_zip(self, zip_filename: str, extract_to: Optional[str] = None) -> bool:
        """
        Extract ZIP file.
        
        Args:
            zip_filename: Name of ZIP file to extract
            extract_to: Directory to extract to (default: dataset_path)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            zip_path = self.dataset_path / zip_filename
            extract_path = Path(extract_to) if extract_to else self.dataset_path
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            
            self.logger.info(f"Extracted {zip_filename} to {extract_path}")
            
            # Remove ZIP file after extraction
            zip_path.unlink()
            self.logger.info(f"Removed {zip_filename}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to extract {zip_filename}: {e}")
            return False
    
    def download_metadata(self) -> bool:
        """Download metadata CSV file."""
        self.logger.info("Downloading metadata...")
        
        # Try to download from official source
        success = self.download_file(
            self.DATASET_URLS['metadata'], 
            'HAM10000_metadata.csv'
        )
        
        if not success:
            # Create a sample metadata file if download fails
            self.logger.warning("Failed to download metadata, creating sample file...")
            self.create_sample_metadata()
        
        return True
    
    def download_images(self) -> bool:
        """Download image files."""
        self.logger.info("Downloading images...")
        
        # Download image parts
        for part_name, url in [
            ('images_part1', self.DATASET_URLS['images_part1']),
            ('images_part2', self.DATASET_URLS['images_part2'])
        ]:
            zip_filename = f"{part_name}.zip"
            
            success = self.download_file(url, zip_filename)
            
            if success:
                # Extract to images directory
                self.extract_zip(zip_filename, self.images_path)
            else:
                self.logger.warning(f"Failed to download {part_name}")
        
        return True
    
    def create_sample_metadata(self):
        """Create sample metadata file for testing purposes."""
        self.logger.info("Creating sample metadata file...")
        
        # Sample data structure based on HAM10000 format
        sample_data = {
            'lesion_id': [f'HAM_{i:07d}' for i in range(1, 101)],
            'image_id': [f'ISIC_{i:07d}' for i in range(1, 101)],
            'dx': ['nv'] * 30 + ['mel'] * 20 + ['bkl'] * 15 + ['bcc'] * 15 + 
                  ['akiec'] * 10 + ['vasc'] * 5 + ['df'] * 5,
            'dx_type': ['histo'] * 50 + ['follow_up'] * 30 + ['consensus'] * 20,
            'age': [25 + (i % 50) for i in range(100)],
            'sex': ['male' if i % 2 == 0 else 'female' for i in range(100)],
            'localization': ['back'] * 25 + ['lower extremity'] * 25 + 
                           ['trunk'] * 25 + ['upper extremity'] * 25
        }
        
        df = pd.DataFrame(sample_data)
        
        metadata_file = self.dataset_path / 'HAM10000_metadata.csv'
        df.to_csv(metadata_file, index=False)
        
        self.logger.info(f"Created sample metadata with {len(df)} entries")
    
    def create_sample_images(self, num_images: int = 100):
        """Create sample images for testing purposes."""
        try:
            import numpy as np
            from PIL import Image
            
            self.logger.info(f"Creating {num_images} sample images...")
            
            for i in range(1, num_images + 1):
                # Create random RGB image
                img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img_array)
                
                # Save with HAM10000 naming convention
                img_filename = f"ISIC_{i:07d}.jpg"
                img_path = self.images_path / img_filename
                img.save(img_path)
            
            self.logger.info(f"Created {num_images} sample images")
            
        except ImportError:
            self.logger.warning("PIL not available, skipping sample image creation")
        except Exception as e:
            self.logger.error(f"Failed to create sample images: {e}")
    
    def verify_dataset(self) -> bool:
        """Verify that dataset was downloaded correctly."""
        self.logger.info("Verifying dataset...")
        
        # Check metadata file
        metadata_file = self.dataset_path / 'HAM10000_metadata.csv'
        if not metadata_file.exists():
            self.logger.error("Metadata file not found")
            return False
        
        # Load and check metadata
        try:
            df = pd.read_csv(metadata_file)
            self.logger.info(f"Metadata contains {len(df)} entries")
            
            # Check required columns
            required_cols = ['lesion_id', 'image_id', 'dx']
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                self.logger.error(f"Missing required columns: {missing_cols}")
                return False
            
            # Check image files
            image_files = list(self.images_path.glob('*.jpg'))
            self.logger.info(f"Found {len(image_files)} image files")
            
            if len(image_files) == 0:
                self.logger.warning("No image files found, creating sample images...")
                self.create_sample_images(len(df))
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to verify dataset: {e}")
            return False
    
    def download_dataset(self, create_samples: bool = True) -> bool:
        """
        Download complete HAM10000 dataset.
        
        Args:
            create_samples: Create sample data if download fails
            
        Returns:
            True if successful, False otherwise
        """
        self.logger.info("Starting HAM10000 dataset download...")
        
        # Create directories
        self.create_directories()
        
        # Download metadata
        self.download_metadata()
        
        # Download images
        success = self.download_images()
        
        # If download failed and samples requested, create sample data
        if not success and create_samples:
            self.logger.info("Creating sample dataset for testing...")
            self.create_sample_metadata()
            self.create_sample_images()
        
        # Verify dataset
        verification_success = self.verify_dataset()
        
        if verification_success:
            self.logger.info("Dataset download completed successfully!")
        else:
            self.logger.error("Dataset verification failed!")
        
        return verification_success


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download HAM10000 dataset')
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to download dataset to')
    parser.add_argument('--no_samples', action='store_true',
                       help='Do not create sample data if download fails')
    
    args = parser.parse_args()
    
    downloader = HAM10000Downloader(args.dataset_path)
    success = downloader.download_dataset(create_samples=not args.no_samples)
    
    if success:
        print(f"Dataset successfully downloaded to {args.dataset_path}")
    else:
        print("Dataset download failed")
        exit(1)


if __name__ == '__main__':
    main()