"""
Предобработка данных для сегментации полей
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import cv2
from sklearn.model_selection import train_test_split
import json


class FieldSegmentationPreprocessor:
    """Препроцессор для данных сегментации полей"""
    
    def __init__(self, data_dir, img_size=(256, 256), batch_size=16):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.batch_size = batch_size
        self.class_names = []
        self.class_to_id = {}
        
    def load_eurosat_data(self):
        """Загрузка EuroSAT датасета и создание масок"""
        print("🔍 Загрузка EuroSAT данных...")
        
        image_paths = []
        labels = []
        
        for class_dir in sorted(self.data_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                if class_name not in self.class_names:
                    self.class_names.append(class_name)
                    self.class_to_id[class_name] = len(self.class_names) - 1
                
                for img_file in class_dir.glob('*.jpg'):
                    image_paths.append(str(img_file))
                    labels.append(class_name)
        
        print(f"✅ Найдено: {len(image_paths)} изображений, {len(self.class_names)} классов")
        return image_paths, labels
    
    def create_synthetic_masks(self, image_paths, labels, output_dir='data/processed/masks'):
        """Создание синтетических масок для классификационного датасета"""
        os.makedirs(output_dir, exist_ok=True)
        
        mask_paths = []
        print("🎨 Создание масок сегментации...")
        
        for img_path, label in zip(image_paths, labels):
            # Создаём маску: всё изображение помечено одним классом
            mask = np.full(self.img_size, self.class_to_id[label], dtype=np.uint8)
            
            # Сохраняем маску
            mask_filename = Path(img_path).stem + '_mask.png'
            mask_path = os.path.join(output_dir, mask_filename)
            Image.fromarray(mask).save(mask_path)
            mask_paths.append(mask_path)
        
        print(f"✅ Создано {len(mask_paths)} масок")
        return mask_paths
    
    def create_dataframe(self, image_paths, labels, mask_paths):
        """Создание DataFrame"""
        df = pd.DataFrame({
            'image_path': image_paths,
            'mask_path': mask_paths,
            'class': labels
        })
        
        print(f"\n📊 Статистика:")
        print(df['class'].value_counts())
        
        return df
    
    def split_data(self, df, test_size=0.15, val_size=0.15, random_state=42):
        """Разделение данных"""
        train_val_df, test_df = train_test_split(
            df, test_size=test_size, random_state=random_state, stratify=df['class']
        )
        
        val_size_adjusted = val_size / (1 - test_size)
        train_df, val_df = train_test_split(
            train_val_df, test_size=val_size_adjusted, 
            random_state=random_state, stratify=train_val_df['class']
        )
        
        print(f"\n✂️  Разделение:")
        print(f"   Train: {len(train_df)}")
        print(f"   Val: {len(val_df)}")
        print(f"   Test: {len(test_df)}")
        
        return train_df, val_df, test_df
    
    def save_config(self, save_path='data/processed/config.json'):
        """Сохранение конфигурации"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        config = {
            'num_classes': len(self.class_names),
            'class_names': self.class_names,
            'class_to_id': self.class_to_id,
            'img_size': self.img_size
        }
        
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        print(f"💾 Конфигурация сохранена: {save_path}")
    
    def prepare_pipeline(self):
        """Полный пайплайн подготовки"""
        print("=" * 70)
        print("🛰️  ПОДГОТОВКА ДАННЫХ ДЛЯ СЕГМЕНТАЦИИ")
        print("=" * 70)
        
        # Загрузка
        image_paths, labels = self.load_eurosat_data()
        
        # Создание масок
        mask_paths = self.create_synthetic_masks(image_paths, labels)
        
        # DataFrame
        df = self.create_dataframe(image_paths, labels, mask_paths)
        
        # Разделение
        train_df, val_df, test_df = self.split_data(df)
        
        # Сохранение конфигурации
        self.save_config()
        
        print("\n" + "=" * 70)
        print("✅ ПОДГОТОВКА ЗАВЕРШЕНА")
        print("=" * 70)
        
        return train_df, val_df, test_df


def load_image_and_mask(image_path, mask_path, img_size=(256, 256)):
    """Загрузка изображения и маски"""
    # Изображение
    img = Image.open(image_path).convert('RGB')
    img = img.resize(img_size)
    img_array = np.array(img) / 255.0
    
    # Маска
    mask = Image.open(mask_path)
    mask = mask.resize(img_size, Image.NEAREST)
    mask_array = np.array(mask)
    
    return img_array, mask_array


if __name__ == "__main__":
    preprocessor = FieldSegmentationPreprocessor(
        data_dir='data/raw/EuroSAT',
        img_size=(256, 256),
        batch_size=16
    )
    
    train_df, val_df, test_df = preprocessor.prepare_pipeline()
    print("\n✅ Пайплайн готов!")
