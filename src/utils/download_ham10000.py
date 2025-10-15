"""
HAM10000 Dataset Downloader using kagglehub library.
"""

import sys
import argparse
import logging
import pandas as pd
from pathlib import Path
import shutil
import os

# Для работы скрипта необходимо сначала установить библиотеку:
# pip install kagglehub
try:
    import kagglehub
except ImportError:
    print("Ошибка: библиотека 'kagglehub' не найдена.")
    print("Пожалуйста, установите ее с помощью команды: pip install kagglehub")
    sys.exit(1)

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)

class HAM10000Downloader:
    """Загрузчик датасета HAM10000 с использованием kagglehub"""
    
    def __init__(self, dataset_path="./dataset"):
        self.dataset_path = Path(dataset_path)
        self.images_dir = self.dataset_path / "images"
        self.metadata_dir = self.dataset_path / "metadata"
        
        # Название датасета в Kaggle
        self.kaggle_dataset_name = "kmader/skin-cancer-mnist-ham10000"
        
        # Ожидаемые колонки в метаданных для проверки
        self.expected_columns = {
            'lesion_id': ['lesion_id'],
            'image_id': ['image_id'],
            'dx': ['dx']
        }
        
        self.create_directories()
    
    def create_directories(self):
        """Создание необходимых директорий"""
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.metadata_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Используется директория для датасета: {self.dataset_path.resolve()}")
    
    def download_with_kagglehub(self):
        """Загрузка и организация датасета через kagglehub"""
        try:
            logger.info(f"Загрузка '{self.kaggle_dataset_name}' с помощью kagglehub...")
            
            # Загружаем датасет через kagglehub
            download_path = kagglehub.dataset_download(self.kaggle_dataset_name)
            logger.info(f"Датасет загружен в: {download_path}")
            
            # Организуем файлы
            self.organize_downloaded_files(download_path)
            
            return True
            
        except Exception as e:
            logger.error(f"Произошла ошибка во время загрузки: {e}")
            return False
    
    def organize_downloaded_files(self, source_dir):
        """Организация файлов: копирование метаданных и изображений."""
        source_path = Path(source_dir)
        logger.info(f"Организация файлов из {source_path}...")
        
        # Ищем CSV файл с метаданными
        metadata_files = list(source_path.rglob("*.csv"))
        if metadata_files:
            # Ищем файл с метаданными HAM10000
            ham_metadata = None
            for metadata_file in metadata_files:
                if "HAM10000" in metadata_file.name or "metadata" in metadata_file.name.lower():
                    ham_metadata = metadata_file
                    break
            
            if not ham_metadata:
                ham_metadata = metadata_files[0]  # Берем первый найденный CSV
            
            target_metadata = self.metadata_dir / "HAM10000_metadata.csv"
            shutil.copy2(str(ham_metadata), str(target_metadata))
            logger.info(f"Метаданные скопированы в {target_metadata}")
        else:
            logger.warning("Файл с метаданными (.csv) не найден.")

        # Ищем все изображения (.jpg) рекурсивно и копируем их
        image_files = list(source_path.rglob("*.jpg"))
        if not image_files:
            logger.warning("Файлы изображений (.jpg) не найдены.")
            return

        copied_count = 0
        for img_file in image_files:
            target_img = self.images_dir / img_file.name
            if not target_img.exists():  # Избегаем дублирования
                shutil.copy2(str(img_file), str(target_img))
                copied_count += 1
        
        logger.info(f"Скопировано {copied_count} изображений в {self.images_dir}")

    def verify_dataset(self):
        """Проверка корректности датасета после загрузки."""
        logger.info("Проверка датасета...")
        
        # Проверка метаданных
        metadata_file = self.metadata_dir / "HAM10000_metadata.csv"
        if not metadata_file.exists():
            logger.error("Файл с метаданными не найден.")
            return False
        
        try:
            df = pd.read_csv(metadata_file)
            # Проверка наличия необходимых колонок
            for col, names in self.expected_columns.items():
                if not any(name in df.columns for name in names):
                    logger.error(f"В метаданных отсутствует необходимая колонка: {col}")
                    return False
        except Exception as e:
            logger.error(f"Ошибка чтения файла с метаданными: {e}")
            return False
        
        # Проверка изображений
        image_files = list(self.images_dir.glob("*.jpg"))
        if not image_files:
            logger.error("Директория с изображениями пуста.")
            return False
        
        logger.info(f"Проверка успешно завершена: найдено {len(df)} записей и {len(image_files)} изображений.")
        return True
    
    def download(self):
        """Основной метод для получения датасета."""
        # Проверяем, есть ли уже валидный датасет
        if self.verify_dataset():
            logger.info("Локальный датасет найден и является валидным.")
            return True
        
        logger.info("Локальный датасет не найден или поврежден. Начинается загрузка...")
        if self.download_with_kagglehub():
            return self.verify_dataset()
        
        return False

def main():
    parser = argparse.ArgumentParser(description="Download HAM10000 dataset using kagglehub.")
    parser.add_argument("--dataset_path", default="./dataset", 
                       help="Путь для сохранения датасета (по умолчанию: ./dataset).")
    parser.add_argument("--check-only", action="store_true",
                       help="Только проверить существующий датасет без загрузки.")
    
    args = parser.parse_args()
    
    downloader = HAM10000Downloader(args.dataset_path)
    
    if args.check_only:
        # Только проверка существующего датасета
        if downloader.verify_dataset():
            print("\n✅ Локальный датасет найден и является валидным.")
            print(f"   Расположение: {downloader.dataset_path.resolve()}")
        else:
            print("\n❌ Локальный датасет не найден или поврежден.")
            sys.exit(1)
    else:
        # Загрузка и проверка датасета
        if downloader.download():
            print("\n✅ Датасет успешно загружен и готов к использованию.")
            print(f"   Расположение: {downloader.dataset_path.resolve()}")
        else:
            print("\n❌ Не удалось загрузить датасет.")
            print("   Проверьте интернет-соединение и убедитесь, что библиотека 'kagglehub' установлена.")
            print("   Также убедитесь, что у вас есть доступ к Kaggle API.")
            sys.exit(1)

if __name__ == "__main__":
    main()