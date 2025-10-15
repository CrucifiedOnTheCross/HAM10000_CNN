"""
Main training script for dermatoscopic image classification using DenseNet.
Supports multiple training scenarios and distributed training on multiple GPUs.
"""

import os
import sys
import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.data.dataset_loader import DatasetLoader
from src.models.densenet_model import DenseNetTransferModel


def setup_logging(experiment_dir: str) -> logging.Logger:
    """Setup logging configuration."""
    log_file = os.path.join(experiment_dir, 'training.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger


def setup_gpu_strategy(logger: logging.Logger) -> tf.distribute.Strategy:
    """Setup distributed training strategy for multiple GPUs."""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    
    if len(gpus) > 1:
        logger.info(f"Found {len(gpus)} GPUs, using MirroredStrategy")
        strategy = tf.distribute.MirroredStrategy()
    else:
        logger.info(f"Found {len(gpus)} GPU(s), using default strategy")
        strategy = tf.distribute.get_strategy()
    
    logger.info(f"Number of replicas: {strategy.num_replicas_in_sync}")
    
    return strategy


def create_experiment_dir(base_dir: str, scenario: str, architecture: str) -> str:
    """Create experiment directory with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"{scenario}_{architecture}_{timestamp}"
    experiment_dir = os.path.join(base_dir, experiment_name)
    
    os.makedirs(experiment_dir, exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(experiment_dir, 'logs'), exist_ok=True)
    
    return experiment_dir


def save_experiment_config(experiment_dir: str, args: argparse.Namespace):
    """Save experiment configuration to JSON file."""
    config = vars(args).copy()
    config['timestamp'] = datetime.now().isoformat()
    
    config_file = os.path.join(experiment_dir, 'config.json')
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)


def create_callbacks(experiment_dir: str, 
                    monitor_metric: str = 'val_f1_score',
                    patience: int = 10) -> list:
    """Create training callbacks."""
    callbacks = []
    
    # Model checkpointing - save best model
    best_model_path = os.path.join(experiment_dir, 'best_model.h5')
    checkpoint_best = tf.keras.callbacks.ModelCheckpoint(
        filepath=best_model_path,
        monitor=monitor_metric,
        mode='max',
        save_best_only=True,
        save_weights_only=False,
        verbose=1
    )
    callbacks.append(checkpoint_best)
    
    # Model checkpointing - save last model
    last_model_path = os.path.join(experiment_dir, 'last_model.h5')
    checkpoint_last = tf.keras.callbacks.ModelCheckpoint(
        filepath=last_model_path,
        save_best_only=False,
        save_weights_only=False,
        verbose=0
    )
    callbacks.append(checkpoint_last)
    
    # Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor=monitor_metric,
        mode='max',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )
    callbacks.append(early_stopping)
    
    # Reduce learning rate on plateau
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor=monitor_metric,
        mode='max',
        factor=0.5,
        patience=patience//2,
        min_lr=1e-7,
        verbose=1
    )
    callbacks.append(reduce_lr)
    
    # TensorBoard logging
    tensorboard_dir = os.path.join(experiment_dir, 'logs')
    tensorboard = tf.keras.callbacks.TensorBoard(
        log_dir=tensorboard_dir,
        histogram_freq=1,
        write_graph=True,
        write_images=True
    )
    callbacks.append(tensorboard)
    
    # CSV logger
    csv_logger = tf.keras.callbacks.CSVLogger(
        os.path.join(experiment_dir, 'training_history.csv')
    )
    callbacks.append(csv_logger)
    
    return callbacks


def calculate_additional_metrics(y_true: np.ndarray, 
                               y_pred: np.ndarray, 
                               y_pred_proba: np.ndarray,
                               class_names: list) -> Dict[str, Any]:
    """Calculate additional metrics beyond what Keras provides."""
    
    # Classification report
    report = classification_report(
        y_true, y_pred, 
        target_names=class_names, 
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # AUC-ROC (one-vs-rest for multiclass)
    y_true_bin = label_binarize(y_true, classes=range(len(class_names)))
    if len(class_names) == 2:
        auc_roc = roc_auc_score(y_true, y_pred_proba[:, 1])
    else:
        auc_roc = roc_auc_score(y_true_bin, y_pred_proba, multi_class='ovr', average='macro')
    
    # Balanced accuracy
    balanced_acc = report['macro avg']['recall']
    
    # Specificity (for each class)
    specificity_per_class = []
    for i in range(len(class_names)):
        tn = np.sum(cm) - (np.sum(cm[i, :]) + np.sum(cm[:, i]) - cm[i, i])
        fp = np.sum(cm[:, i]) - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_per_class.append(specificity)
    
    avg_specificity = np.mean(specificity_per_class)
    
    metrics = {
        'classification_report': report,
        'confusion_matrix': cm.tolist(),
        'auc_roc': float(auc_roc),
        'balanced_accuracy': float(balanced_acc),
        'specificity_per_class': specificity_per_class,
        'average_specificity': float(avg_specificity)
    }
    
    return metrics


def evaluate_model(model: tf.keras.Model,
                  test_dataset: tf.data.Dataset,
                  class_names: list,
                  experiment_dir: str,
                  logger: logging.Logger) -> Dict[str, Any]:
    """Evaluate model on test set and save results."""
    
    logger.info("Evaluating model on test set...")
    
    # Get predictions
    y_true = []
    y_pred_proba = []
    
    for batch_x, batch_y in test_dataset:
        y_true.extend(batch_y.numpy())
        pred_proba = model.predict(batch_x, verbose=0)
        y_pred_proba.extend(pred_proba)
    
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    test_metrics = calculate_additional_metrics(y_true, y_pred, y_pred_proba, class_names)
    
    # Save results
    results_file = os.path.join(experiment_dir, 'test_results.json')
    with open(results_file, 'w') as f:
        json.dump(test_metrics, f, indent=2)
    
    # Log key metrics
    logger.info(f"Test Results:")
    logger.info(f"  Accuracy: {test_metrics['classification_report']['accuracy']:.4f}")
    logger.info(f"  Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
    logger.info(f"  Macro F1-Score: {test_metrics['classification_report']['macro avg']['f1-score']:.4f}")
    logger.info(f"  AUC-ROC: {test_metrics['auc_roc']:.4f}")
    logger.info(f"  Average Specificity: {test_metrics['average_specificity']:.4f}")
    
    return test_metrics


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train DenseNet for dermatoscopic image classification')
    
    # Dataset arguments
    parser.add_argument('--dataset_path', type=str, required=True,
                       help='Path to HAM10000 dataset directory')
    parser.add_argument('--metadata_file', type=str, default='HAM10000_metadata.csv',
                       help='Metadata CSV filename')
    
    # Model arguments
    parser.add_argument('--architecture', type=str, choices=['densenet121', 'densenet201'], 
                       default='densenet121', help='DenseNet architecture')
    parser.add_argument('--scenario', type=str, 
                       choices=['head_only', 'partial_unfreeze', 'full_training'],
                       default='head_only', help='Training scenario')
    parser.add_argument('--unfreeze_percent', type=float, default=20.0,
                       help='Percentage of top layers to unfreeze for partial_unfreeze scenario')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate (auto-selected based on scenario if not provided)')
    parser.add_argument('--dropout_rate', type=float, default=0.5, help='Dropout rate')
    
    # Data processing arguments
    parser.add_argument('--image_size', type=int, default=224, help='Input image size')
    parser.add_argument('--augmentation', action='store_true', help='Enable data augmentation')
    parser.add_argument('--use_class_weights', action='store_true', help='Use class weights for imbalanced data')
    
    # Experiment arguments
    parser.add_argument('--experiment_dir', type=str, default='experiments',
                       help='Base directory for experiments')
    parser.add_argument('--early_stopping_patience', type=int, default=10,
                       help='Early stopping patience')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Create experiment directory
    experiment_dir = create_experiment_dir(args.experiment_dir, args.scenario, args.architecture)
    
    # Setup logging
    logger = setup_logging(experiment_dir)
    logger.info(f"Starting experiment: {os.path.basename(experiment_dir)}")
    
    # Save experiment configuration
    save_experiment_config(experiment_dir, args)
    
    # Setup GPU strategy
    strategy = setup_gpu_strategy(logger)
    
    # Load and prepare dataset
    logger.info("Loading dataset...")
    dataset_loader = DatasetLoader(
        dataset_path=args.dataset_path,
        metadata_file=args.metadata_file,
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size
    )
    
    # Download dataset if needed
    if not os.path.exists(os.path.join(args.dataset_path, args.metadata_file)):
        logger.info("Dataset not found, downloading...")
        dataset_loader.download_dataset()
    
    # Load metadata and create datasets
    metadata = dataset_loader.load_metadata()
    train_ds, val_ds, test_ds = dataset_loader.create_datasets(
        augment_train=args.augmentation
    )
    
    # Get class information
    class_names = dataset_loader.get_class_names()
    num_classes = len(class_names)
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Class names: {class_names}")
    
    # Calculate class weights if requested
    class_weights = None
    if args.use_class_weights:
        class_weights = dataset_loader.calculate_class_weights()
        logger.info(f"Using class weights: {class_weights}")
    
    # Build model within strategy scope
    with strategy.scope():
        # Create model
        model_builder = DenseNetTransferModel(
            num_classes=num_classes,
            input_shape=(args.image_size, args.image_size, 3),
            architecture=args.architecture,
            dropout_rate=args.dropout_rate
        )
        
        # Build and configure model
        model = model_builder.build_model(args.scenario)
        
        # Configure training scenario with unfreeze percentage
        if args.scenario == 'partial_unfreeze':
            model_builder.configure_training_scenario(args.scenario, args.unfreeze_percent)
        
        # Compile model
        model = model_builder.compile_model(
            scenario=args.scenario,
            learning_rate=args.learning_rate,
            class_weights=class_weights
        )
    
    # Log model information
    model_info = model_builder.get_layer_info()
    logger.info(f"Model info: {model_info}")
    
    # Save model summary
    summary_file = os.path.join(experiment_dir, 'model_summary.txt')
    with open(summary_file, 'w') as f:
        f.write(model_builder.get_model_summary())
    
    # Create callbacks
    callbacks = create_callbacks(experiment_dir, patience=args.early_stopping_patience)
    
    # Train model
    logger.info("Starting training...")
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    # Save training history
    history_file = os.path.join(experiment_dir, 'training_history.json')
    with open(history_file, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        history_dict = {k: [float(x) for x in v] for k, v in history.history.items()}
        json.dump(history_dict, f, indent=2)
    
    # Load best model for evaluation
    best_model_path = os.path.join(experiment_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        logger.info("Loading best model for evaluation...")
        model = tf.keras.models.load_model(best_model_path)
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_ds, class_names, experiment_dir, logger)
    
    logger.info(f"Experiment completed successfully!")
    logger.info(f"Results saved in: {experiment_dir}")


if __name__ == '__main__':
    main()