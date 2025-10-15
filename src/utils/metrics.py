"""
Utility functions for calculating and logging model performance metrics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from typing import Dict, List, Tuple, Any
import os
import json


class MetricsCalculator:
    """Calculate comprehensive metrics for multi-class classification."""
    
    def __init__(self, class_names: List[str]):
        """
        Initialize metrics calculator.
        
        Args:
            class_names: List of class names
        """
        self.class_names = class_names
        self.num_classes = len(class_names)
    
    def calculate_all_metrics(self, 
                            y_true: np.ndarray, 
                            y_pred: np.ndarray, 
                            y_pred_proba: np.ndarray) -> Dict[str, Any]:
        """
        Calculate all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=self.class_names, 
            output_dict=True,
            zero_division=0
        )
        metrics['classification_report'] = report
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm.tolist()
        
        # Per-class metrics
        metrics['per_class_metrics'] = self._calculate_per_class_metrics(cm)
        
        # AUC-ROC
        metrics['auc_roc'] = self._calculate_auc_roc(y_true, y_pred_proba)
        
        # Balanced accuracy
        metrics['balanced_accuracy'] = report['macro avg']['recall']
        
        # Overall accuracy
        metrics['accuracy'] = report['accuracy']
        
        # Macro averages
        metrics['macro_precision'] = report['macro avg']['precision']
        metrics['macro_recall'] = report['macro avg']['recall']
        metrics['macro_f1'] = report['macro avg']['f1-score']
        
        # Weighted averages
        metrics['weighted_precision'] = report['weighted avg']['precision']
        metrics['weighted_recall'] = report['weighted avg']['recall']
        metrics['weighted_f1'] = report['weighted avg']['f1-score']
        
        return metrics
    
    def _calculate_per_class_metrics(self, cm: np.ndarray) -> Dict[str, List[float]]:
        """Calculate per-class precision, recall, specificity, and F1-score."""
        per_class = {
            'precision': [],
            'recall': [],
            'specificity': [],
            'f1_score': []
        }
        
        for i in range(self.num_classes):
            # True positives, false positives, false negatives, true negatives
            tp = cm[i, i]
            fp = np.sum(cm[:, i]) - tp
            fn = np.sum(cm[i, :]) - tp
            tn = np.sum(cm) - (tp + fp + fn)
            
            # Precision
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            per_class['precision'].append(precision)
            
            # Recall (Sensitivity)
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            per_class['recall'].append(recall)
            
            # Specificity
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            per_class['specificity'].append(specificity)
            
            # F1-score
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            per_class['f1_score'].append(f1)
        
        return per_class
    
    def _calculate_auc_roc(self, y_true: np.ndarray, y_pred_proba: np.ndarray) -> Dict[str, float]:
        """Calculate AUC-ROC for multi-class classification."""
        auc_metrics = {}
        
        if self.num_classes == 2:
            # Binary classification
            auc_metrics['binary'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            auc_metrics['macro'] = auc_metrics['binary']
            auc_metrics['weighted'] = auc_metrics['binary']
        else:
            # Multi-class classification
            y_true_bin = label_binarize(y_true, classes=range(self.num_classes))
            
            # Macro average (one-vs-rest)
            try:
                auc_metrics['macro'] = roc_auc_score(
                    y_true_bin, y_pred_proba, 
                    multi_class='ovr', average='macro'
                )
            except ValueError:
                auc_metrics['macro'] = 0.0
            
            # Weighted average
            try:
                auc_metrics['weighted'] = roc_auc_score(
                    y_true_bin, y_pred_proba, 
                    multi_class='ovr', average='weighted'
                )
            except ValueError:
                auc_metrics['weighted'] = 0.0
            
            # Per-class AUC
            per_class_auc = []
            for i in range(self.num_classes):
                try:
                    auc_score = roc_auc_score(y_true_bin[:, i], y_pred_proba[:, i])
                    per_class_auc.append(auc_score)
                except ValueError:
                    per_class_auc.append(0.0)
            
            auc_metrics['per_class'] = per_class_auc
        
        return auc_metrics
    
    def save_metrics_report(self, 
                          metrics: Dict[str, Any], 
                          output_dir: str, 
                          filename: str = 'metrics_report.json'):
        """Save metrics to JSON file."""
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
    
    def create_confusion_matrix_plot(self, 
                                   cm: np.ndarray, 
                                   output_dir: str,
                                   filename: str = 'confusion_matrix.png',
                                   normalize: bool = True):
        """Create and save confusion matrix plot."""
        plt.figure(figsize=(10, 8))
        
        if normalize:
            cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Normalized Confusion Matrix')
        else:
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=self.class_names, yticklabels=self.class_names)
            plt.title('Confusion Matrix')
        
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_metrics_summary_plot(self, 
                                  metrics: Dict[str, Any], 
                                  output_dir: str,
                                  filename: str = 'metrics_summary.png'):
        """Create summary plot of key metrics."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Per-class precision, recall, F1-score
        per_class = metrics['per_class_metrics']
        x_pos = np.arange(len(self.class_names))
        
        axes[0, 0].bar(x_pos - 0.2, per_class['precision'], 0.2, label='Precision', alpha=0.8)
        axes[0, 0].bar(x_pos, per_class['recall'], 0.2, label='Recall', alpha=0.8)
        axes[0, 0].bar(x_pos + 0.2, per_class['f1_score'], 0.2, label='F1-Score', alpha=0.8)
        axes[0, 0].set_xlabel('Classes')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Per-Class Metrics')
        axes[0, 0].set_xticks(x_pos)
        axes[0, 0].set_xticklabels(self.class_names, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Per-class specificity
        axes[0, 1].bar(x_pos, per_class['specificity'], alpha=0.8, color='green')
        axes[0, 1].set_xlabel('Classes')
        axes[0, 1].set_ylabel('Specificity')
        axes[0, 1].set_title('Per-Class Specificity')
        axes[0, 1].set_xticks(x_pos)
        axes[0, 1].set_xticklabels(self.class_names, rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # Overall metrics comparison
        overall_metrics = {
            'Accuracy': metrics['accuracy'],
            'Balanced Accuracy': metrics['balanced_accuracy'],
            'Macro Precision': metrics['macro_precision'],
            'Macro Recall': metrics['macro_recall'],
            'Macro F1': metrics['macro_f1']
        }
        
        metric_names = list(overall_metrics.keys())
        metric_values = list(overall_metrics.values())
        
        axes[1, 0].bar(metric_names, metric_values, alpha=0.8, color='orange')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Overall Metrics')
        axes[1, 0].set_xticklabels(metric_names, rotation=45)
        axes[1, 0].grid(True, alpha=0.3)
        
        # AUC-ROC per class (if available)
        if 'per_class' in metrics['auc_roc']:
            auc_scores = metrics['auc_roc']['per_class']
            axes[1, 1].bar(x_pos, auc_scores, alpha=0.8, color='red')
            axes[1, 1].set_xlabel('Classes')
            axes[1, 1].set_ylabel('AUC-ROC')
            axes[1, 1].set_title('Per-Class AUC-ROC')
            axes[1, 1].set_xticks(x_pos)
            axes[1, 1].set_xticklabels(self.class_names, rotation=45)
            axes[1, 1].grid(True, alpha=0.3)
        else:
            # Show macro AUC-ROC
            axes[1, 1].bar(['Macro AUC-ROC'], [metrics['auc_roc']['macro']], 
                          alpha=0.8, color='red')
            axes[1, 1].set_ylabel('AUC-ROC')
            axes[1, 1].set_title('AUC-ROC Score')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_metrics_summary(self, metrics: Dict[str, Any]):
        """Print formatted metrics summary to console."""
        print("\n" + "="*60)
        print("CLASSIFICATION METRICS SUMMARY")
        print("="*60)
        
        print(f"\nOverall Metrics:")
        print(f"  Accuracy:           {metrics['accuracy']:.4f}")
        print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
        print(f"  Macro Precision:    {metrics['macro_precision']:.4f}")
        print(f"  Macro Recall:       {metrics['macro_recall']:.4f}")
        print(f"  Macro F1-Score:     {metrics['macro_f1']:.4f}")
        
        if 'macro' in metrics['auc_roc']:
            print(f"  Macro AUC-ROC:      {metrics['auc_roc']['macro']:.4f}")
        
        print(f"\nPer-Class Metrics:")
        per_class = metrics['per_class_metrics']
        
        print(f"{'Class':<15} {'Precision':<10} {'Recall':<10} {'F1-Score':<10} {'Specificity':<12}")
        print("-" * 60)
        
        for i, class_name in enumerate(self.class_names):
            print(f"{class_name:<15} "
                  f"{per_class['precision'][i]:<10.4f} "
                  f"{per_class['recall'][i]:<10.4f} "
                  f"{per_class['f1_score'][i]:<10.4f} "
                  f"{per_class['specificity'][i]:<12.4f}")
        
        print("\n" + "="*60)


def create_training_plots(history_dict: Dict[str, List[float]], 
                         output_dir: str,
                         filename: str = 'training_history.png'):
    """Create training history plots."""
    
    # Determine available metrics
    metrics_to_plot = []
    if 'accuracy' in history_dict and 'val_accuracy' in history_dict:
        metrics_to_plot.append(('accuracy', 'val_accuracy', 'Accuracy'))
    if 'loss' in history_dict and 'val_loss' in history_dict:
        metrics_to_plot.append(('loss', 'val_loss', 'Loss'))
    if 'f1_score' in history_dict and 'val_f1_score' in history_dict:
        metrics_to_plot.append(('f1_score', 'val_f1_score', 'F1-Score'))
    if 'precision' in history_dict and 'val_precision' in history_dict:
        metrics_to_plot.append(('precision', 'val_precision', 'Precision'))
    if 'recall' in history_dict and 'val_recall' in history_dict:
        metrics_to_plot.append(('recall', 'val_recall', 'Recall'))
    
    if not metrics_to_plot:
        print("No suitable metrics found for plotting")
        return
    
    # Create subplots
    n_metrics = len(metrics_to_plot)
    cols = 2
    rows = (n_metrics + 1) // 2
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (train_metric, val_metric, title) in enumerate(metrics_to_plot):
        ax = axes[i] if n_metrics > 1 else axes
        
        epochs = range(1, len(history_dict[train_metric]) + 1)
        
        ax.plot(epochs, history_dict[train_metric], 'b-', label=f'Training {title}')
        ax.plot(epochs, history_dict[val_metric], 'r-', label=f'Validation {title}')
        
        ax.set_title(f'Model {title}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename), dpi=300, bbox_inches='tight')
    plt.close()