"""
Оценка модели сегментации
"""

import os
import sys
import argparse
import json
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

sys.path.append('src')

from model import SegmentationDataGenerator
import segmentation_models as sm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/best_model.h5')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--num_samples', type=int, default=10)
    return parser.parse_args()


def visualize_segmentation(images, true_masks, pred_masks, class_names, save_path):
    """Визуализация результатов сегментации"""
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    
    if n == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(n):
        # Оригинальное изображение
        axes[i, 0].imshow(images[i])
        axes[i, 0].set_title('Original Image')
        axes[i, 0].axis('off')
        
        # Ground truth
        true_class = np.argmax(true_masks[i], axis=-1)
        axes[i, 1].imshow(true_class, cmap='tab10')
        axes[i, 1].set_title('Ground Truth')
        axes[i, 1].axis('off')
        
        # Предсказание
        pred_class = np.argmax(pred_masks[i], axis=-1)
        axes[i, 2].imshow(pred_class, cmap='tab10')
        axes[i, 2].set_title('Prediction')
        axes[i, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"✅ Визуализация сохранена: {save_path}")
    plt.close()


def main():
    args = parse_args()
    
    print("=" * 80)
    print("📊 ОЦЕНКА МОДЕЛИ СЕГМЕНТАЦИИ")
    print("=" * 80)
    
    # Загрузка модели
    print(f"\n📥 Загрузка модели: {args.model_path}")
    model = tf.keras.models.load_model(
        args.model_path,
        custom_objects={
            'dice_loss_plus_1categorical_focal_loss': sm.losses.DiceLoss() + sm.losses.CategoricalFocalLoss(),
            'iou_score': sm.metrics.IOUScore(),
            'f1-score': sm.metrics.FScore()
        }
    )
    print("✅ Модель загружена")
    
    # Загрузка конфигурации
    with open('models/config.json', 'r') as f:
        config = json.load(f)
    
    # Загрузка тестовых данных
    import pandas as pd
    test_df = pd.read_csv('data/processed/test.csv')
    
    test_gen = SegmentationDataGenerator(
        test_df, batch_size=8, img_size=(256, 256),
        num_classes=config['num_classes'], augment=False
    )
    
    # Оценка
    print("\n🔮 Оценка на тестовой выборке...")
    results = model.evaluate(test_gen, verbose=1)
    
    print("\n" + "=" * 80)
    print("📊 РЕЗУЛЬТАТЫ")
    print("=" * 80)
    print(f"\nTest Loss: {results[0]:.4f}")
    print(f"Test IoU Score: {results[1]:.4f}")
    print(f"Test F1-Score: {results[2]:.4f}")
    print(f"Test Accuracy: {results[3]:.4f}")
    
    # Визуализация
    if args.visualize:
        print("\n📊 Создание визуализаций...")
        images, masks = next(iter(test_gen))
        predictions = model.predict(images[:args.num_samples])
        
        visualize_segmentation(
            images[:args.num_samples],
            masks[:args.num_samples],
            predictions,
            config['class_names'],
            'reports/segmentation_results.png'
        )
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
