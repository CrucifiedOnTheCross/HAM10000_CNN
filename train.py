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
import matplotlib.pyplot as plt
import seaborn as sns
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
                               y_pred_proba: np.ndarray,
                               class_names: list) -> Dict[str, Any]:
    """Calculate additional metrics beyond what Keras provides."""
    
    # Convert probabilities to class predictions
    y_pred = np.argmax(y_pred_proba, axis=1)
    
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


def plot_training_metrics(history: tf.keras.callbacks.History, 
                         experiment_dir: str,
                         logger: logging.Logger) -> None:
    """
    Plot training metrics over epochs using seaborn.
    
    Args:
        history: Training history from model.fit()
        experiment_dir: Directory to save plots
        logger: Logger instance
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8')
    
    # Get metrics from history
    metrics = history.history
    epochs = range(1, len(metrics['loss']) + 1)
    
    # Create plots directory
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Define metrics to plot
    metric_pairs = [
        ('loss', 'val_loss', 'Loss'),
        ('accuracy', 'val_accuracy', 'Accuracy'),
        ('precision', 'val_precision', 'Precision'),
        ('recall', 'val_recall', 'Recall'),
        ('f1_score', 'val_f1_score', 'F1 Score')
    ]
    
    # Create subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Training Metrics Over Time', fontsize=16, fontweight='bold')
    
    # Flatten axes for easier iteration
    axes = axes.flatten()
    
    for idx, (train_metric, val_metric, title) in enumerate(metric_pairs):
        if train_metric in metrics and val_metric in metrics:
            ax = axes[idx]
            
            # Plot training and validation metrics
            ax.plot(epochs, metrics[train_metric], 'o-', label=f'Training {title}', 
                   linewidth=2, markersize=4)
            ax.plot(epochs, metrics[val_metric], 's-', label=f'Validation {title}', 
                   linewidth=2, markersize=4)
            
            ax.set_title(f'{title} Over Epochs', fontsize=12, fontweight='bold')
            ax.set_xlabel('Epoch')
            ax.set_ylabel(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Add best value annotation
            best_val = max(metrics[val_metric]) if 'loss' not in val_metric else min(metrics[val_metric])
            best_epoch = (np.argmax(metrics[val_metric]) if 'loss' not in val_metric 
                         else np.argmin(metrics[val_metric])) + 1
            
            ax.annotate(f'Best: {best_val:.4f}\n(Epoch {best_epoch})',
                       xy=(best_epoch, best_val),
                       xytext=(0.7, 0.95), textcoords='axes fraction',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                       fontsize=9)
    
    # Remove empty subplot
    if len(metric_pairs) < len(axes):
        fig.delaxes(axes[-1])
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'training_metrics.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Training metrics plot saved to: {plot_path}")


def plot_test_results(metrics: Dict[str, Any], 
                     experiment_dir: str,
                     class_names: list,
                     logger: logging.Logger) -> None:
    """
    Create visualizations for test set results.
    
    Args:
        metrics: Dictionary containing test metrics
        experiment_dir: Directory to save plots
        class_names: List of class names
        logger: Logger instance
    """
    # Set seaborn style
    sns.set_style("whitegrid")
    plt.style.use('seaborn-v0_8')
    
    # Create plots directory
    plots_dir = os.path.join(experiment_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Confusion Matrix Heatmap
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Test Set Results Visualization', fontsize=16, fontweight='bold')
    
    # Confusion Matrix
    ax1 = axes[0, 0]
    cm = metrics['confusion_matrix']
    # Convert to numpy array if it's a list
    if isinstance(cm, list):
        cm = np.array(cm)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=ax1)
    ax1.set_title('Confusion Matrix', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Predicted Label')
    ax1.set_ylabel('True Label')
    
    # Normalized Confusion Matrix
    ax2 = axes[0, 1]
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names, ax=ax2)
    ax2.set_title('Normalized Confusion Matrix', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Predicted Label')
    ax2.set_ylabel('True Label')
    
    # Per-class metrics bar plot
    ax3 = axes[1, 0]
    if 'classification_report_dict' in metrics:
        report = metrics['classification_report_dict']
        classes = [cls for cls in report.keys() if cls not in ['accuracy', 'macro avg', 'weighted avg']]
        
        precision_scores = [report[cls]['precision'] for cls in classes]
        recall_scores = [report[cls]['recall'] for cls in classes]
        f1_scores = [report[cls]['f1-score'] for cls in classes]
        
        x = np.arange(len(classes))
        width = 0.25
        
        ax3.bar(x - width, precision_scores, width, label='Precision', alpha=0.8)
        ax3.bar(x, recall_scores, width, label='Recall', alpha=0.8)
        ax3.bar(x + width, f1_scores, width, label='F1-Score', alpha=0.8)
        
        ax3.set_xlabel('Classes')
        ax3.set_ylabel('Score')
        ax3.set_title('Per-Class Metrics', fontsize=12, fontweight='bold')
        ax3.set_xticks(x)
        ax3.set_xticklabels(classes, rotation=45)
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Overall metrics summary
    ax4 = axes[1, 1]
    overall_metrics = {
        'Accuracy': metrics.get('accuracy', 0),
        'Balanced Accuracy': metrics.get('balanced_accuracy', 0),
        'AUC-ROC': metrics.get('auc_roc', 0),
        'Avg Specificity': metrics.get('average_specificity', 0)
    }
    
    metric_names = list(overall_metrics.keys())
    metric_values = list(overall_metrics.values())
    
    bars = ax4.bar(metric_names, metric_values, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
    ax4.set_title('Overall Test Metrics', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Score')
    ax4.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, value in zip(bars, metric_values):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(plots_dir, 'test_results.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Test results visualization saved to: {plot_path}")
    
    # Create separate specificity plot if available
    if 'specificity_per_class' in metrics:
        plt.figure(figsize=(10, 6))
        specificity_values = list(metrics['specificity_per_class'].values())
        
        bars = plt.bar(class_names, specificity_values, color='lightsteelblue', alpha=0.8)
        plt.title('Specificity per Class', fontsize=14, fontweight='bold')
        plt.xlabel('Classes')
        plt.ylabel('Specificity')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, specificity_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        # Save specificity plot
        specificity_plot_path = os.path.join(plots_dir, 'specificity_per_class.png')
        plt.savefig(specificity_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Specificity per class plot saved to: {specificity_plot_path}")


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
    
    # Convert one-hot encoded y_true to integer labels for sklearn metrics
    if y_true.ndim > 1 and y_true.shape[1] > 1:
        y_true = np.argmax(y_true, axis=1)
    
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Calculate metrics
    test_metrics = calculate_additional_metrics(y_true, y_pred_proba, class_names)
    
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
        image_size=(args.image_size, args.image_size),
        batch_size=args.batch_size
    )
    
    # Download dataset if needed
    if not os.path.exists(os.path.join(args.dataset_path, 'metadata', args.metadata_file)):
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
    callbacks = create_callbacks(experiment_dir, monitor_metric='val_f1_score', patience=args.early_stopping_patience)
    
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
    
    # Plot training metrics
    logger.info("Creating training metrics visualization...")
    plot_training_metrics(history, experiment_dir, logger)
    
    # Load best model for evaluation
    best_model_path = os.path.join(experiment_dir, 'best_model.h5')
    if os.path.exists(best_model_path):
        logger.info("Loading best model for evaluation...")
        model = tf.keras.models.load_model(best_model_path)
    
    # Evaluate on test set
    test_results = evaluate_model(model, test_ds, class_names, experiment_dir, logger)
    
    # Plot test results visualization
    logger.info("Creating test results visualization...")
    plot_test_results(test_results, experiment_dir, class_names, logger)
    
    logger.info(f"Experiment completed successfully!")
    logger.info(f"Results saved in: {experiment_dir}")


if __name__ == '__main__':
    main()