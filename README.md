# Dermatoscopic Image Classification using DenseNet

Система классификации дерматоскопических изображений на основе DenseNet с использованием transfer learning для датасета HAM10000.

## Описание проекта

Этот проект реализует систему классификации кожных поражений на 7 типов с использованием сверточных нейронных сетей DenseNet и техник transfer learning. Система поддерживает различные сценарии обучения, распределенное обучение на нескольких GPU и комплексную систему метрик.

### Классы кожных поражений (HAM10000):
- **nv** - Melanocytic nevi (меланоцитарные невусы)
- **mel** - Melanoma (меланома)
- **bkl** - Benign keratosis-like lesions (доброкачественные кератозоподобные поражения)
- **bcc** - Basal cell carcinoma (базально-клеточная карцинома)
- **akiec** - Actinic keratoses and intraepithelial carcinoma (актинические кератозы)
- **vasc** - Vascular lesions (сосудистые поражения)
- **df** - Dermatofibroma (дерматофиброма)

## Структура проекта

```
TransferLearning/
├── src/
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset_loader.py      # Загрузка и предобработка данных
│   ├── models/
│   │   ├── __init__.py
│   │   └── densenet_model.py      # Модели DenseNet с transfer learning
│   └── utils/
│       ├── __init__.py
│       ├── metrics.py             # Расчет метрик и визуализация
│       └── download_ham10000.py   # Скачивание датасета
├── dataset/
│   ├── images/                    # Изображения HAM10000
│   └── metadata/                  # Метаданные
├── experiments/                   # Результаты экспериментов
├── configs/                       # Конфигурационные файлы
├── train.py                       # Основной скрипт обучения
├── task.md                        # Техническое задание
└── README.md                      # Документация
```

## Требования

### Системные требования:
- Python 3.8+
- TensorFlow 2.x
- CUDA-совместимая GPU (рекомендуется)
- Минимум 8GB RAM
- Минимум 10GB свободного места на диске

### Python зависимости:
```bash
pip install tensorflow>=2.8.0
pip install numpy pandas scikit-learn
pip install matplotlib seaborn
pip install Pillow tqdm requests
```

## Установка и настройка

1. **Клонирование проекта:**
```bash
git clone <repository_url>
cd TransferLearning
```

2. **Установка зависимостей:**
```bash
pip install tensorflow numpy pandas scikit-learn matplotlib seaborn Pillow tqdm requests
```

3. **Скачивание датасета:**
```bash
python src/utils/download_ham10000.py --dataset_path ./dataset
```

## Использование

### Основные параметры командной строки

```bash
python train.py --dataset_path ./dataset [OPTIONS]
```

#### Обязательные параметры:
- `--dataset_path` - Путь к директории с датасетом HAM10000

#### Параметры модели:
- `--architecture` - Архитектура DenseNet (`densenet121`, `densenet201`)
- `--scenario` - Сценарий обучения (`head_only`, `partial_unfreeze`, `full_training`)
- `--unfreeze_percent` - Процент верхних слоев для разморозки (для `partial_unfreeze`)
- `--dropout_rate` - Коэффициент dropout (по умолчанию: 0.5)

#### Параметры обучения:
- `--batch_size` - Размер батча (по умолчанию: 32)
- `--epochs` - Количество эпох (по умолчанию: 50)
- `--learning_rate` - Скорость обучения (автоматически выбирается по сценарию)
- `--early_stopping_patience` - Терпение для early stopping (по умолчанию: 10)

#### Параметры данных:
- `--image_size` - Размер входных изображений (по умолчанию: 224)
- `--augmentation` - Включить аугментацию данных
- `--use_class_weights` - Использовать веса классов для несбалансированных данных

#### Параметры эксперимента:
- `--experiment_dir` - Директория для сохранения результатов (по умолчанию: experiments)
- `--metadata_file` - Имя файла метаданных (по умолчанию: HAM10000_metadata.csv)

### Примеры использования

#### 1. Обучение только головы классификатора (рекомендуется для начала):
```bash
python train.py \
    --dataset_path ./dataset \
    --scenario head_only \
    --architecture densenet121 \
    --batch_size 32 \
    --epochs 30 \
    --augmentation \
    --use_class_weights
```

#### 2. Частичное размораживание верхних слоев:
```bash
python train.py \
    --dataset_path ./dataset \
    --scenario partial_unfreeze \
    --unfreeze_percent 25 \
    --architecture densenet201 \
    --batch_size 16 \
    --epochs 50 \
    --learning_rate 0.0001 \
    --augmentation \
    --use_class_weights
```

#### 3. Полное обучение всей модели:
```bash
python train.py \
    --dataset_path ./dataset \
    --scenario full_training \
    --architecture densenet121 \
    --batch_size 8 \
    --epochs 100 \
    --learning_rate 0.00001 \
    --augmentation \
    --use_class_weights \
    --early_stopping_patience 15
```

#### 4. Быстрый тест с минимальными параметрами:
```bash
python train.py \
    --dataset_path ./dataset \
    --scenario head_only \
    --epochs 5 \
    --batch_size 16
```

## Сценарии обучения

### 1. Head Only (`head_only`)
- **Описание:** Обучается только кастомная голова классификатора
- **Замороженные слои:** Все слои базовой модели DenseNet
- **Рекомендуемая LR:** 0.001
- **Время обучения:** Быстро (~30-60 мин)
- **Использование:** Первичная настройка, быстрое прототипирование

### 2. Partial Unfreeze (`partial_unfreeze`)
- **Описание:** Размораживаются верхние слои базовой модели + голова
- **Настраиваемые слои:** Верхние N% слоев (параметр `--unfreeze_percent`)
- **Рекомендуемая LR:** 0.0001
- **Время обучения:** Средне (~1-3 часа)
- **Использование:** Тонкая настройка после head_only

### 3. Full Training (`full_training`)
- **Описание:** Обучается вся модель с низкой скоростью обучения
- **Настраиваемые слои:** Все слои модели
- **Рекомендуемая LR:** 0.00001
- **Время обучения:** Долго (~3-8 часов)
- **Использование:** Максимальная производительность

## Результаты и метрики

### Автоматически рассчитываемые метрики:
- **Accuracy** - Общая точность
- **Balanced Accuracy** - Сбалансированная точность
- **Precision** - Точность (macro/weighted/per-class)
- **Recall** - Полнота (macro/weighted/per-class)
- **F1-Score** - F1-мера (macro/weighted/per-class)
- **Specificity** - Специфичность (per-class)
- **AUC-ROC** - Площадь под ROC-кривой
- **Confusion Matrix** - Матрица ошибок

### Сохраняемые файлы результатов:
```
experiments/scenario_architecture_timestamp/
├── config.json                 # Конфигурация эксперимента
├── training.log               # Логи обучения
├── model_summary.txt          # Архитектура модели
├── best_model.h5             # Лучшая модель по F1-score
├── last_model.h5             # Последняя модель
├── training_history.csv      # История обучения
├── training_history.json     # История обучения (JSON)
├── test_results.json         # Результаты на тестовой выборке
├── checkpoints/              # Промежуточные чекпоинты
└── logs/                     # TensorBoard логи
```

### Визуализация результатов:
- **TensorBoard:** `tensorboard --logdir experiments/experiment_name/logs`
- **Графики обучения:** Автоматически сохраняются в директории эксперимента
- **Матрица ошибок:** Визуализация и численные значения

## Распределенное обучение

Система автоматически определяет доступные GPU и использует `tf.distribute.MirroredStrategy` для обучения на нескольких GPU:

```python
# Автоматическое определение стратегии
gpus = tf.config.experimental.list_physical_devices('GPU')
if len(gpus) > 1:
    strategy = tf.distribute.MirroredStrategy()
else:
    strategy = tf.distribute.get_strategy()
```

## Рекомендации по использованию

### Последовательность экспериментов:
1. **Начните с `head_only`** для быстрой проверки работоспособности
2. **Перейдите к `partial_unfreeze`** для улучшения результатов
3. **Используйте `full_training`** для максимальной производительности

### Настройка гиперпараметров:
- **Batch size:** Уменьшите при нехватке GPU памяти
- **Learning rate:** Используйте автоматические значения или настройте вручную
- **Epochs:** Используйте early stopping для предотвращения переобучения
- **Augmentation:** Всегда включайте для улучшения генерализации

### Мониторинг обучения:
```bash
# Запуск TensorBoard
tensorboard --logdir experiments/

# Просмотр логов в реальном времени
tail -f experiments/experiment_name/training.log
```

## Устранение неполадок

### Частые проблемы:

1. **OutOfMemoryError:**
   - Уменьшите `batch_size`
   - Используйте `mixed_precision` обучение
   - Закройте другие GPU-приложения

2. **Датасет не найден:**
   - Проверьте путь к датасету
   - Запустите скрипт скачивания: `python src/utils/download_ham10000.py`

3. **Медленное обучение:**
   - Проверьте использование GPU: `nvidia-smi`
   - Увеличьте `batch_size` при наличии памяти
   - Используйте `tf.data` оптимизации

4. **Низкое качество модели:**
   - Увеличьте количество эпох
   - Включите аугментацию данных
   - Используйте веса классов для несбалансированных данных
   - Попробуйте другой сценарий обучения

### Логирование и отладка:
```bash
# Включить детальное логирование TensorFlow
export TF_CPP_MIN_LOG_LEVEL=0

# Проверить доступные GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

## Лицензия

Этот проект предназначен для образовательных и исследовательских целей.

## Контакты

При возникновении вопросов или проблем создайте issue в репозитории проекта.