"""
DenseNet model builder for transfer learning on HAM10000 dataset.
Supports different training scenarios: head_only, partial_unfreeze, full_training.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import DenseNet121, DenseNet201
from typing import Tuple, Optional, Dict, Any
import logging

# Импорты для дополнительных метрик
try:
    import tensorflow_addons as tfa
    TFA_AVAILABLE = True
except ImportError:
    TFA_AVAILABLE = False
    logging.warning("tensorflow_addons не установлен. F1Score метрика будет недоступна.")


class F1Score(tf.keras.metrics.Metric):
    """Custom F1 Score metric for multiclass classification."""
    
    def __init__(self, num_classes, average='macro', name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.average = average
        
        # Initialize true positives, false positives, false negatives
        self.true_positives = self.add_weight(
            name='tp', shape=(num_classes,), initializer='zeros'
        )
        self.false_positives = self.add_weight(
            name='fp', shape=(num_classes,), initializer='zeros'
        )
        self.false_negatives = self.add_weight(
            name='fn', shape=(num_classes,), initializer='zeros'
        )
    
    def update_state(self, y_true, y_pred, sample_weight=None):
        # Convert predictions to class indices
        y_pred = tf.argmax(y_pred, axis=1)
        y_true = tf.argmax(y_true, axis=1)
        
        # Convert to one-hot for per-class calculations
        y_true_onehot = tf.one_hot(y_true, self.num_classes)
        y_pred_onehot = tf.one_hot(y_pred, self.num_classes)
        
        # Calculate true positives, false positives, false negatives
        tp = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
        fp = tf.reduce_sum((1 - y_true_onehot) * y_pred_onehot, axis=0)
        fn = tf.reduce_sum(y_true_onehot * (1 - y_pred_onehot), axis=0)
        
        self.true_positives.assign_add(tp)
        self.false_positives.assign_add(fp)
        self.false_negatives.assign_add(fn)
    
    def result(self):
        # Calculate precision and recall for each class
        precision = self.true_positives / (self.true_positives + self.false_positives + tf.keras.backend.epsilon())
        recall = self.true_positives / (self.true_positives + self.false_negatives + tf.keras.backend.epsilon())
        
        # Calculate F1 score for each class
        f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
        
        if self.average == 'macro':
            return tf.reduce_mean(f1)
        elif self.average == 'weighted':
            # For weighted average, we'd need class weights
            return tf.reduce_mean(f1)
        else:
            return f1
    
    def reset_state(self):
        self.true_positives.assign(tf.zeros_like(self.true_positives))
        self.false_positives.assign(tf.zeros_like(self.false_positives))
        self.false_negatives.assign(tf.zeros_like(self.false_negatives))


class DenseNetTransferModel:
    """
    DenseNet-based transfer learning model for dermatoscopic image classification.
    
    Supports three training scenarios:
    1. head_only: Only train the custom classification head
    2. partial_unfreeze: Unfreeze top layers of base model + head
    3. full_training: Train entire model with low learning rate
    """
    
    def __init__(self, 
                 num_classes: int = 7,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 architecture: str = 'densenet121',
                 dropout_rate: float = 0.5):
        """
        Initialize DenseNet transfer learning model.
        
        Args:
            num_classes: Number of output classes
            input_shape: Input image shape (height, width, channels)
            architecture: DenseNet architecture ('densenet121' or 'densenet201')
            dropout_rate: Dropout rate for regularization
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.architecture = architecture.lower()
        self.dropout_rate = dropout_rate
        
        self.base_model = None
        self.model = None
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Validate architecture
        if self.architecture not in ['densenet121', 'densenet201']:
            raise ValueError(f"Unsupported architecture: {architecture}")
    
    def create_base_model(self, weights: str = 'imagenet') -> tf.keras.Model:
        """
        Create base DenseNet model with pretrained weights.
        
        Args:
            weights: Pretrained weights ('imagenet' or None)
            
        Returns:
            Base DenseNet model
        """
        if self.architecture == 'densenet121':
            base_model = DenseNet121(
                weights=weights,
                include_top=False,
                input_shape=self.input_shape
            )
        else:  # densenet201
            base_model = DenseNet201(
                weights=weights,
                include_top=False,
                input_shape=self.input_shape
            )
        
        self.base_model = base_model
        self.logger.info(f"Created {self.architecture} base model with {weights} weights")
        self.logger.info(f"Base model has {len(base_model.layers)} layers")
        
        return base_model
    
    def create_classification_head(self, base_model: tf.keras.Model) -> tf.keras.Model:
        """
        Create custom classification head on top of base model.
        
        Args:
            base_model: Base DenseNet model
            
        Returns:
            Complete model with classification head
        """
        # Global average pooling
        x = layers.GlobalAveragePooling2D()(base_model.output)
        
        # Dense layer with ReLU activation
        x = layers.Dense(512, activation='relu', name='dense_512')(x)
        x = layers.BatchNormalization(name='bn_512')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_512')(x)
        
        # Optional second dense layer
        x = layers.Dense(256, activation='relu', name='dense_256')(x)
        x = layers.BatchNormalization(name='bn_256')(x)
        x = layers.Dropout(self.dropout_rate, name='dropout_256')(x)
        
        # Output layer
        predictions = layers.Dense(
            self.num_classes, 
            activation='softmax', 
            name='predictions'
        )(x)
        
        # Create complete model
        model = Model(inputs=base_model.input, outputs=predictions)
        
        self.model = model
        self.logger.info(f"Created complete model with {len(model.layers)} layers")
        
        return model
    
    def build_model(self, scenario: str = 'head_only') -> tf.keras.Model:
        """
        Build complete model for specified training scenario.
        
        Args:
            scenario: Training scenario ('head_only', 'partial_unfreeze', 'full_training')
            
        Returns:
            Complete model ready for training
        """
        # Create base model
        base_model = self.create_base_model()
        
        # Create complete model with classification head
        model = self.create_classification_head(base_model)
        
        # Configure trainable layers based on scenario
        self.configure_training_scenario(scenario)
        
        return model
    
    def configure_training_scenario(self, scenario: str, unfreeze_percent: float = 20.0):
        """
        Configure which layers are trainable based on training scenario.
        
        Args:
            scenario: Training scenario
            unfreeze_percent: Percentage of top layers to unfreeze for partial_unfreeze
        """
        if self.base_model is None or self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        scenario = scenario.lower()
        
        if scenario == 'head_only':
            # Freeze all base model layers
            self.base_model.trainable = False
            trainable_params = sum([layer.count_params() 
                                  for layer in self.model.layers if layer.trainable])
            self.logger.info(f"HEAD_ONLY: Frozen base model, {trainable_params:,} trainable parameters")
            
        elif scenario == 'partial_unfreeze':
            # Freeze base model initially
            self.base_model.trainable = True
            
            # Calculate number of layers to unfreeze
            total_layers = len(self.base_model.layers)
            unfreeze_layers = int(total_layers * unfreeze_percent / 100)
            freeze_layers = total_layers - unfreeze_layers
            
            # Freeze bottom layers
            for layer in self.base_model.layers[:freeze_layers]:
                layer.trainable = False
            
            # Unfreeze top layers
            for layer in self.base_model.layers[freeze_layers:]:
                layer.trainable = True
            
            trainable_params = sum([layer.count_params() 
                                  for layer in self.model.layers if layer.trainable])
            self.logger.info(f"PARTIAL_UNFREEZE: Unfroze top {unfreeze_percent}% ({unfreeze_layers}) layers, "
                           f"{trainable_params:,} trainable parameters")
            
        elif scenario == 'full_training':
            # Unfreeze all layers
            self.base_model.trainable = True
            trainable_params = sum([layer.count_params() 
                                  for layer in self.model.layers if layer.trainable])
            self.logger.info(f"FULL_TRAINING: All layers trainable, {trainable_params:,} trainable parameters")
            
        else:
            raise ValueError(f"Unknown training scenario: {scenario}")
    
    def compile_model(self, 
                     scenario: str,
                     learning_rate: Optional[float] = None,
                     class_weights: Optional[Dict[int, float]] = None) -> tf.keras.Model:
        """
        Compile model with appropriate optimizer and loss function.
        
        Args:
            scenario: Training scenario (affects default learning rate)
            learning_rate: Custom learning rate (overrides defaults)
            class_weights: Class weights for imbalanced dataset
            
        Returns:
            Compiled model
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        # Set default learning rates based on scenario
        if learning_rate is None:
            lr_defaults = {
                'head_only': 0.001,
                'partial_unfreeze': 0.0001,
                'full_training': 0.00001
            }
            learning_rate = lr_defaults.get(scenario.lower(), 0.001)
        
        # Create optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        
        # Создаем список базовых метрик
        metrics_list = [
            'accuracy',
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy')
        ]
        
        # Добавляем F1Score метрику (используем tensorflow_addons если доступен, иначе кастомную)
        if TFA_AVAILABLE:
            metrics_list.append(tfa.metrics.F1Score(num_classes=self.num_classes, average='macro', name='f1_score'))
        else:
            metrics_list.append(F1Score(num_classes=self.num_classes, average='macro', name='f1_score'))
            self.logger.info("Используется кастомная F1Score метрика")
        
        # Compile model
        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics_list
        )
        
        self.logger.info(f"Compiled model for {scenario} with learning rate {learning_rate}")
        
        return self.model
    
    def get_model_summary(self) -> str:
        """
        Get model summary as string.
        
        Returns:
            Model summary string
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        import io
        import sys
        
        # Capture summary output
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        
        self.model.summary()
        
        sys.stdout = old_stdout
        summary = buffer.getvalue()
        
        return summary
    
    def save_model(self, filepath: str, save_format: str = 'tf'):
        """
        Save model to file.
        
        Args:
            filepath: Path to save model
            save_format: Format to save ('tf' for SavedModel, 'h5' for HDF5)
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        if save_format.lower() == 'h5':
            self.model.save(f"{filepath}.h5")
            self.logger.info(f"Model saved as HDF5: {filepath}.h5")
        else:
            self.model.save(filepath)
            self.logger.info(f"Model saved as SavedModel: {filepath}")
    
    def load_model(self, filepath: str) -> tf.keras.Model:
        """
        Load model from file.
        
        Args:
            filepath: Path to model file
            
        Returns:
            Loaded model
        """
        self.model = tf.keras.models.load_model(filepath)
        self.logger.info(f"Model loaded from: {filepath}")
        
        return self.model
    
    def get_layer_info(self) -> Dict[str, Any]:
        """
        Get information about model layers.
        
        Returns:
            Dictionary with layer information
        """
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        
        info = {
            'total_layers': len(self.model.layers),
            'trainable_layers': sum([1 for layer in self.model.layers if layer.trainable]),
            'total_params': self.model.count_params(),
            'trainable_params': sum([layer.count_params() 
                                   for layer in self.model.layers if layer.trainable]),
            'base_model_layers': len(self.base_model.layers) if self.base_model else 0
        }
        
        info['non_trainable_params'] = info['total_params'] - info['trainable_params']
        
        return info