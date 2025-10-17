"""
Simple CNN model for fast baseline experiments on HAM10000.
Implements the same interface as DenseNetTransferModel to integrate with train.py.
"""

import logging
from typing import Tuple, Optional, Dict, Any

import tensorflow as tf
from tensorflow.keras import layers, Model

# Reuse F1Score metric from DenseNet model
try:
    from src.models.densenet_model import F1Score
except Exception:
    # Fallback: basic F1Score-like metric (macro) if import fails
    class F1Score(tf.keras.metrics.Metric):
        def __init__(self, num_classes, average='macro', name='f1_score', **kwargs):
            super().__init__(name=name, **kwargs)
            self.num_classes = num_classes
            self.average = average
            self.tp = self.add_weight(shape=(num_classes,), initializer='zeros', name='tp')
            self.fp = self.add_weight(shape=(num_classes,), initializer='zeros', name='fp')
            self.fn = self.add_weight(shape=(num_classes,), initializer='zeros', name='fn')

        def update_state(self, y_true, y_pred, sample_weight=None):
            y_pred = tf.argmax(y_pred, axis=1)
            y_true = tf.argmax(y_true, axis=1)
            y_true_onehot = tf.one_hot(y_true, self.num_classes)
            y_pred_onehot = tf.one_hot(y_pred, self.num_classes)
            tp = tf.reduce_sum(y_true_onehot * y_pred_onehot, axis=0)
            fp = tf.reduce_sum((1 - y_true_onehot) * y_pred_onehot, axis=0)
            fn = tf.reduce_sum(y_true_onehot * (1 - y_pred_onehot), axis=0)
            self.tp.assign_add(tp)
            self.fp.assign_add(fp)
            self.fn.assign_add(fn)

        def result(self):
            precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
            recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
            f1 = 2 * (precision * recall) / (precision + recall + tf.keras.backend.epsilon())
            return tf.reduce_mean(f1)


class SimpleCNNModel:
    """
    Lightweight CNN baseline with a small convolutional trunk and a classification head.
    Supports scenarios: head_only, partial_unfreeze, full_training.
    """

    def __init__(self,
                 num_classes: int = 7,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 dropout_rate: float = 0.5):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.dropout_rate = dropout_rate
        self.base_model: Optional[Model] = None
        self.model: Optional[Model] = None

        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

    def create_base_model(self) -> tf.keras.Model:
        """Create a simple convolutional feature extractor (base model)."""
        inputs = layers.Input(shape=self.input_shape)
        x = inputs

        # Block 1
        x = layers.Conv2D(32, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        # Block 2
        x = layers.Conv2D(64, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.MaxPooling2D()(x)

        # Block 3
        x = layers.Conv2D(128, 3, padding='same', activation=None)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)

        base_outputs = layers.GlobalAveragePooling2D()(x)
        base_model = tf.keras.Model(inputs=inputs, outputs=base_outputs, name='simple_cnn_base')
        self.base_model = base_model
        self.logger.info("Created SimpleCNN base model")
        return base_model

    def create_classification_head(self, base_model: tf.keras.Model) -> tf.keras.Model:
        """Attach a small classification head on top of base_model."""
        x = base_model.output
        x = layers.Dense(128, activation='relu')(x)
        x = layers.Dropout(self.dropout_rate)(x)
        outputs = layers.Dense(self.num_classes, activation='softmax')(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs, name='simple_cnn_classifier')
        self.model = model
        self.logger.info("Attached classification head to SimpleCNN base")
        return model

    def build_model(self, scenario: str = 'head_only') -> tf.keras.Model:
        base_model = self.create_base_model()
        model = self.create_classification_head(base_model)
        self.configure_training_scenario(scenario)
        return model

    def configure_training_scenario(self, scenario: str, unfreeze_percent: float = 20.0):
        """Configure trainable layers for scenarios analogous to DenseNet model."""
        if self.base_model is None or self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        scenario = scenario.lower()
        if scenario == 'head_only':
            # Freeze convolutional trunk, train only the head
            self.base_model.trainable = False
            self.logger.info("HEAD_ONLY: Frozen base conv trunk")
        elif scenario == 'partial_unfreeze':
            self.base_model.trainable = True
            total_layers = len(self.base_model.layers)
            unfreeze_layers = int(total_layers * unfreeze_percent / 100)
            freeze_layers = total_layers - unfreeze_layers
            for layer in self.base_model.layers[:freeze_layers]:
                layer.trainable = False
            for layer in self.base_model.layers[freeze_layers:]:
                layer.trainable = True
            self.logger.info(f"PARTIAL_UNFREEZE: Unfroze top {unfreeze_percent}% layers of base conv trunk")
        elif scenario == 'full_training':
            self.base_model.trainable = True
            self.logger.info("FULL_TRAINING: All layers trainable")
        else:
            raise ValueError(f"Unknown training scenario: {scenario}")

    def compile_model(self,
                      scenario: str,
                      learning_rate: Optional[float] = None,
                      class_weights: Optional[Dict[int, float]] = None) -> tf.keras.Model:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")

        if learning_rate is None:
            lr_defaults = {
                'head_only': 0.001,
                'partial_unfreeze': 0.0005,
                'full_training': 0.0003
            }
            learning_rate = lr_defaults.get(scenario.lower(), 0.001)

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        metrics_list = [
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.TopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
            F1Score(num_classes=self.num_classes, average='macro', name='f1_score')
        ]

        self.model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=metrics_list
        )
        self.logger.info(f"Compiled SimpleCNN for {scenario} with learning rate {learning_rate}")
        return self.model

    def get_model_summary(self) -> str:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        import io
        import sys
        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()
        self.model.summary()
        sys.stdout = old_stdout
        return buffer.getvalue()

    def get_layer_info(self) -> Dict[str, Any]:
        if self.model is None:
            raise ValueError("Model not built. Call build_model() first.")
        info = {
            'total_layers': len(self.model.layers),
            'trainable_layers': sum([1 for layer in self.model.layers if layer.trainable]),
            'total_params': self.model.count_params(),
            'trainable_params': sum([layer.count_params() for layer in self.model.layers if layer.trainable]),
            'base_model_layers': len(self.base_model.layers) if self.base_model else 0
        }
        info['non_trainable_params'] = info['total_params'] - info['trainable_params']
        return info