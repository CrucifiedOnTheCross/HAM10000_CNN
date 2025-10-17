"""
DatasetLoader class for HAM10000 dermatoscopic image dataset.
Handles loading, preprocessing, augmentation and data splitting.
"""

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import cv2
from typing import Tuple, Dict, Optional
import logging


class DatasetLoader:
    """
    Dataset loader for HAM10000 dermatoscopic images.
    
    Handles:
    - Loading metadata and images
    - Image preprocessing and normalization
    - Data augmentation
    - Stratified train/validation/test splitting
    - Class weight computation for imbalanced dataset
    - tf.data.Dataset preparation
    """
    
    def __init__(self, 
                 dataset_path: str,
                 image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 random_state: int = 42):
        """
        Initialize DatasetLoader.
        
        Args:
            dataset_path: Path to dataset directory
            image_size: Target image size (height, width)
            batch_size: Batch size for tf.data.Dataset
            random_state: Random state for reproducibility
        """
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.batch_size = batch_size
        self.random_state = random_state
        
        # HAM10000 class names
        self.class_names = [
            'akiec',  # Actinic keratoses and intraepithelial carcinoma
            'bcc',    # Basal cell carcinoma
            'bkl',    # Benign keratosis-like lesions
            'df',     # Dermatofibroma
            'mel',    # Melanoma
            'nv',     # Melanocytic nevi
            'vasc'    # Vascular lesions
        ]
        
        self.label_encoder = LabelEncoder()
        self.metadata = None
        self.class_weights = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def load_metadata(self) -> pd.DataFrame:
        """
        Load HAM10000 metadata from CSV file.
        
        Returns:
            DataFrame with metadata
        """
        metadata_path = os.path.join(self.dataset_path, 'metadata', 'HAM10000_metadata.csv')
        
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        self.metadata = pd.read_csv(metadata_path)
        self.logger.info(f"Loaded metadata: {len(self.metadata)} samples")
        
        # Encode labels
        self.metadata['label_encoded'] = self.label_encoder.fit_transform(self.metadata['dx'])
        
        # Log class distribution
        class_counts = self.metadata['dx'].value_counts()
        self.logger.info("Class distribution:")
        for class_name, count in class_counts.items():
            self.logger.info(f"  {class_name}: {count} samples")
            
        return self.metadata
    
    def compute_class_weights(self) -> Dict[int, float]:
        """
        Compute class weights for imbalanced dataset.
        
        Returns:
            Dictionary mapping class indices to weights
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
            
        # Compute class weights
        class_weights_array = compute_class_weight(
            'balanced',
            classes=np.unique(self.metadata['label_encoded']),
            y=self.metadata['label_encoded']
        )
        
        self.class_weights = {i: weight for i, weight in enumerate(class_weights_array)}
        
        self.logger.info("Computed class weights:")
        for i, weight in self.class_weights.items():
            class_name = self.label_encoder.inverse_transform([i])[0]
            self.logger.info(f"  {class_name}: {weight:.3f}")
            
        return self.class_weights
    
    def calculate_class_weights(self) -> Dict[int, float]:
        """
        Alias for compute_class_weights to match train.py usage.
        """
        return self.compute_class_weights()
    
    def load_and_preprocess_image(self, image_path: str) -> tf.Tensor:
        """
        Load and preprocess a single image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor
        """
        # Load image
        image = tf.io.read_file(image_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image = tf.cast(image, tf.float32)
        
        # Resize to target size
        image = tf.image.resize(image, self.image_size)
        
        # Normalize to [0, 1]
        image = image / 255.0
        
        return image
    
    def create_augmentation_layer(self) -> tf.keras.Sequential:
        """
        Create data augmentation layer.
        
        Returns:
            Sequential model with augmentation layers
        """
        augmentation = tf.keras.Sequential([
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomZoom(0.1),
            tf.keras.layers.RandomContrast(0.1),
            tf.keras.layers.RandomBrightness(0.1),
            # Add Gaussian noise
            tf.keras.layers.GaussianNoise(0.01)
        ])
        
        return augmentation
    
    def split_data(self, test_size: float = 0.15, val_size: float = 0.15) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Split data into train/validation/test sets with stratification.
        
        Args:
            test_size: Proportion of test set
            val_size: Proportion of validation set
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if self.metadata is None:
            raise ValueError("Metadata not loaded. Call load_metadata() first.")
            
        # First split: train+val vs test
        train_val_df, test_df = train_test_split(
            self.metadata,
            test_size=test_size,
            stratify=self.metadata['dx'],
            random_state=self.random_state
        )
        
        # Second split: train vs val
        val_size_adjusted = val_size / (1 - test_size)  # Adjust for remaining data
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_size_adjusted,
            stratify=train_val_df['dx'],
            random_state=self.random_state
        )
        
        self.logger.info(f"Data split - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def create_tf_dataset(self, 
                         df: pd.DataFrame, 
                         augment: bool = False,
                         shuffle: bool = True) -> tf.data.Dataset:
        """
        Create tf.data.Dataset from DataFrame.
        
        Args:
            df: DataFrame with image paths and labels
            augment: Whether to apply data augmentation
            shuffle: Whether to shuffle the dataset
            
        Returns:
            tf.data.Dataset
        """
        # Get image paths and labels
        image_paths = [os.path.join(self.dataset_path, 'images', f"{img_id}.jpg") 
                      for img_id in df['image_id']]
        labels = df['label_encoded'].values
        
        # Convert labels to one-hot encoding
        num_classes = len(self.class_names)
        labels_one_hot = tf.keras.utils.to_categorical(labels, num_classes=num_classes)
        
        # Create dataset from paths and labels
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels_one_hot))
        
        # Map preprocessing function
        dataset = dataset.map(
            lambda path, label: (self.load_and_preprocess_image(path), label),
            num_parallel_calls=tf.data.AUTOTUNE
        )
        
        # Apply augmentation if requested
        if augment:
            augmentation_layer = self.create_augmentation_layer()
            dataset = dataset.map(
                lambda image, label: (augmentation_layer(image, training=True), label),
                num_parallel_calls=tf.data.AUTOTUNE
            )
        
        # Shuffle if requested
        if shuffle:
            dataset = dataset.shuffle(buffer_size=1000, seed=self.random_state)
        
        # Batch and prefetch
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        
        return dataset
    
    def prepare_datasets(self, 
                        augment_train: bool = True,
                        test_size: float = 0.15,
                        val_size: float = 0.15) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Prepare train, validation and test datasets.
        
        Args:
            augment_train: Whether to augment training data
            test_size: Proportion of test set
            val_size: Proportion of validation set
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        # Load metadata if not already loaded
        if self.metadata is None:
            self.load_metadata()
        
        # Split data
        train_df, val_df, test_df = self.split_data(test_size, val_size)
        
        # Create datasets
        train_dataset = self.create_tf_dataset(train_df, augment=augment_train, shuffle=True)
        val_dataset = self.create_tf_dataset(val_df, augment=False, shuffle=False)
        test_dataset = self.create_tf_dataset(test_df, augment=False, shuffle=False)
        
        self.logger.info("Datasets prepared successfully")
        
        return train_dataset, val_dataset, test_dataset
    
    def create_datasets(self, 
                       augment_train: bool = True,
                       test_size: float = 0.15,
                       val_size: float = 0.15) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """
        Create train, validation and test datasets.
        This is an alias for prepare_datasets method for compatibility.
        
        Args:
            augment_train: Whether to augment training data
            test_size: Proportion of test set
            val_size: Proportion of validation set
            
        Returns:
            Tuple of (train_dataset, val_dataset, test_dataset)
        """
        return self.prepare_datasets(augment_train, test_size, val_size)
    
    def get_class_names(self) -> list:
        """
        Get list of class names.
        
        Returns:
            List of class names
        """
        return self.class_names
    
    def get_dataset_info(self) -> Dict:
        """
        Get information about the dataset.
        
        Returns:
            Dictionary with dataset information
        """
        if self.metadata is None:
            self.load_metadata()
            
        info = {
            'total_samples': len(self.metadata),
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'image_size': self.image_size,
            'batch_size': self.batch_size,
            'class_distribution': self.metadata['dx'].value_counts().to_dict()
        }
        
        return info
    
    def download_dataset(self) -> bool:
        """
        Download or verify HAM10000 dataset using HAM10000Downloader.
        Returns True if dataset is ready.
        """
        try:
            from src.utils.download_ham10000 import HAM10000Downloader
        except Exception as e:
            self.logger.error(f"Failed to import downloader: {e}")
            return False
        
        downloader = HAM10000Downloader(self.dataset_path)
        ready = downloader.download()
        if ready:
            self.logger.info("Dataset is ready for use.")
            return True
        else:
            self.logger.error("Failed to download/verify dataset.")
            return False