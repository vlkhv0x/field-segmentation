"""
Скрипт обучения модели сегментации
"""

import os
import sys
import argparse
import json
import numpy as np

sys.path.append('src')

from data_preprocessing import FieldSegmentationPreprocessor
from model import UNetSegmentation, SegmentationDataGenerator


def parse_args():
    parser = argparse.ArgumentParser(description='Обучение модели сегментации')
    parser.add_argument('--data_dir', type=str, default='data/raw/EuroSAT')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--encoder', type=str, default='resnet34',
                       choices=['resnet34', 'resnet50', 'efficientnetb3'])
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("🛰️  ОБУЧЕНИЕ МОДЕЛИ СЕГМЕНТАЦИИ ПОЛЕЙ")
    print("=" * 80)
    
    # Подготовка данных
    if not os.path.exists('data/processed/config.json'):
        preprocessor = FieldSegmentationPreprocessor(
            data_dir=args.data_dir,
            img_size=(256, 256),
            batch_size=args.batch_size
        )
        train_df, val_df, test_df = preprocessor.prepare_pipeline()
    else:
        import pandas as pd
        print("📂 Загрузка существующих данных...")
        train_df = pd.read_csv('data/processed/train.csv')
        val_df = pd.read_csv('data/processed/val.csv')
        test_df = pd.read_csv('data/processed/test.csv')
    
    # Загрузка конфигурации
    with open('data/processed/config.json', 'r') as f:
        config = json.load(f)
    
    num_classes = config['num_classes']
    
    # Создание генераторов
    print("\n🔄 Создание генераторов данных...")
    train_gen = SegmentationDataGenerator(
        train_df, args.batch_size, (256, 256), num_classes, augment=True
    )
    val_gen = SegmentationDataGenerator(
        val_df, args.batch_size, (256, 256), num_classes, augment=False
    )
    
    # Создание модели
    print("\n🏗️  Создание U-Net модели...")
    model_builder = UNetSegmentation(
        num_classes=num_classes,
        img_size=(256, 256),
        encoder=args.encoder
    )
    
    model = model_builder.build_model()
    model_builder.compile_model(learning_rate=args.learning_rate)
    
    print(f"✅ Модель создана: {args.encoder}")
    print(f"   Параметров: {model.count_params():,}")
    
    # Обучение
    history = model_builder.train(
        train_gen=train_gen,
        val_gen=val_gen,
        epochs=args.epochs
    )
    
    # Сохранение
    model_builder.save_model('models/final_model.h5')
    
    # Сохранение истории
    history_dict = {k: [float(v) for v in vals] 
                   for k, vals in history.history.items()}
    with open('reports/training_history.json', 'w') as f:
        json.dump(history_dict, f, indent=4)
    
    # Сохранение конфигурации модели
    model_config = {
        'encoder': args.encoder,
        'num_classes': num_classes,
        'img_size': 256,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'class_names': config['class_names']
    }
    
    with open('models/config.json', 'w') as f:
        json.dump(model_config, f, indent=4)
    
    print("\n" + "=" * 80)
    print("✅ ОБУЧЕНИЕ ЗАВЕРШЕНО")
    print("=" * 80)
    
    best_iou = max(history.history['val_iou_score'])
    best_fscore = max(history.history['val_f1-score'])
    
    print(f"\n📈 Лучшие результаты:")
    print(f"   Val IoU Score: {best_iou:.4f}")
    print(f"   Val F1-Score: {best_fscore:.4f}")
    
    print(f"\n🎯 Следующий шаг:")
    print("   python src/evaluate.py")


if __name__ == "__main__":
    main()
